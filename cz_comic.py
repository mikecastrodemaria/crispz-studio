"""crispz-studio - comic: modele de projet BD, layouts de planche, casting.

Couche PUREMENT geometrique et documentaire: aucun import torch / cz_pipeline, donc
importable et testable sans GPU (les tests tournent en <1s). Le moteur (txt2img,
omni, inpaint) est appele par l'appelant -- ce module lui dit QUOI generer, a
QUELLE taille, avec QUELLES references, et recompose la planche apres coup.

Contient:
  - LAYOUTS / PAGE_PRESETS : gabarits de planche (fractions) et formats de page
  - panel_rects / gen_size : geometrie des cases -> pixels, et taille de generation
  - casting : resolution des @Name -> description + refs Omni + LoRA
  - projet : new_project / load_project / save_project / add_chapter / add_page
  - compose_page / export_pdf : assemblage de la planche finale

Principe important (rappel du bug corrige plusieurs fois ici): un FRAGMENT ne recoit
jamais le prompt de la scene. La passe de detail sur un crop passe par detail_prompt(),
qui renvoie une description LOCALE, jamais le texte du panneau. Voir detail_prompt().
"""

import os
import re
import json
import math

from PIL import Image, ImageDraw, ImageOps

SCHEMA_VERSION = 1

# Tolerance sur les fractions de layout (un bord a 0.5 doit etre reconnu comme
# interieur, un bord a 1.0 comme exterieur, malgre les flottants).
_EPS = 1e-6

# Alignement des dimensions de generation. 32 et pas 16: le transformer Z-Image
# patchifie par 2 le latent VAE (voir cz_pipeline.round_to_multiple).
GEN_ALIGN = 32


# ----------------------------------------------------------------------------
# Gabarits
# ----------------------------------------------------------------------------
# Une case = (x, y, w, h) en FRACTIONS de la zone utile (page moins les marges).
# L'ordre de la liste = l'ordre de lecture des cases.
LAYOUTS = {
    "splash":       [(0, 0, 1, 1)],
    "2-up":         [(0, 0, 1, .5), (0, .5, 1, .5)],
    "2-side":       [(0, 0, .5, 1), (.5, 0, .5, 1)],
    "3-classic":    [(0, 0, 1, .4), (0, .4, .5, .6), (.5, .4, .5, .6)],
    "3-strip":      [(0, 0, 1, 1 / 3), (0, 1 / 3, 1, 1 / 3), (0, 2 / 3, 1, 1 / 3)],
    "4-grid":       [(0, 0, .5, .5), (.5, 0, .5, .5), (0, .5, .5, .5), (.5, .5, .5, .5)],
    "4-wide-top":   [(0, 0, 1, .4), (0, .4, 1 / 3, .6), (1 / 3, .4, 1 / 3, .6),
                     (2 / 3, .4, 1 / 3, .6)],
    "5-hero":       [(0, 0, 1, .45), (0, .45, .5, .275), (.5, .45, .5, .275),
                     (0, .725, .5, .275), (.5, .725, .5, .275)],
    "6-grid":       [(0, 0, .5, 1 / 3), (.5, 0, .5, 1 / 3),
                     (0, 1 / 3, .5, 1 / 3), (.5, 1 / 3, .5, 1 / 3),
                     (0, 2 / 3, .5, 1 / 3), (.5, 2 / 3, .5, 1 / 3)],
    "9-grid":       [(c / 3, r / 3, 1 / 3, 1 / 3) for r in range(3) for c in range(3)],
}

# Formats de planche courants. dpi sert a l'export (PDF) et a rien d'autre.
PAGE_PRESETS = {
    "A4 300dpi":        {"width": 2480, "height": 3508, "dpi": 300},
    "US comic 300dpi":  {"width": 1988, "height": 3075, "dpi": 300},   # 6.625 x 10.25 in
    "A4 150dpi":        {"width": 1240, "height": 1754, "dpi": 150},
    "Web":              {"width": 1280, "height": 1980, "dpi": 96},
}

DEFAULT_PAGE = {
    "width": 2480, "height": 3508, "dpi": 300,
    "margin": 96,          # blanc autour de la zone utile (px)
    "gutter": 48,          # gouttiere ENTRE deux cases (px)
    "background": "#ffffff",
    "border": 0,           # cadre noir autour de chaque case (px, 0 = aucun)
    "border_color": "#000000",
}

PANEL_STATUS = ("draft", "locked")


def layout_names():
    return sorted(LAYOUTS)


def layout_cells(name):
    """Cases d'un gabarit. Leve ValueError si le nom est inconnu."""
    cells = LAYOUTS.get(name)
    if cells is None:
        raise ValueError(f"unknown layout '{name}' (known: {', '.join(layout_names())})")
    return list(cells)


def validate_cells(cells):
    """Verifie qu'un gabarit tient dans [0,1] et que ses cases ne se chevauchent pas.
    Renvoie la liste des problemes (vide = gabarit sain)."""
    problems = []
    for i, cell in enumerate(cells):
        if len(cell) != 4:
            problems.append(f"cell {i}: expected 4 values, got {len(cell)}")
            continue
        x, y, w, h = cell
        if w <= 0 or h <= 0:
            problems.append(f"cell {i}: non-positive size {w}x{h}")
        if x < -_EPS or y < -_EPS or x + w > 1 + _EPS or y + h > 1 + _EPS:
            problems.append(f"cell {i}: out of the unit square ({x},{y},{w},{h})")
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            ax, ay, aw, ah = cells[i]
            bx, by, bw, bh = cells[j]
            ox = min(ax + aw, bx + bw) - max(ax, bx)
            oy = min(ay + ah, by + bh) - max(ay, by)
            if ox > _EPS and oy > _EPS:
                problems.append(f"cells {i} and {j} overlap")
    return problems


# ----------------------------------------------------------------------------
# Geometrie
# ----------------------------------------------------------------------------
def panel_rects(cells, page_w, page_h, margin=0, gutter=0):
    """Fractions -> rectangles pixel (x, y, w, h) sur la planche.

    Chaque case est rentree de gutter/2 sur ses bords INTERIEURS uniquement: deux
    cases voisines sont donc separees d'exactement `gutter`, et les cases de bord
    touchent exactement la marge (pas de demi-gouttiere parasite au bord de page)."""
    cw = page_w - 2 * margin
    ch = page_h - 2 * margin
    if cw <= 0 or ch <= 0:
        raise ValueError(f"margin {margin} too large for a {page_w}x{page_h} page")
    g = gutter / 2.0
    out = []
    for idx, (fx, fy, fw, fh) in enumerate(cells):
        x0 = margin + fx * cw
        y0 = margin + fy * ch
        x1 = margin + (fx + fw) * cw
        y1 = margin + (fy + fh) * ch
        if fx > _EPS:
            x0 += g
        if fy > _EPS:
            y0 += g
        if fx + fw < 1 - _EPS:
            x1 -= g
        if fy + fh < 1 - _EPS:
            y1 -= g
        x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"cell {idx} collapsed: gutter {gutter} / margin {margin} "
                             f"too large for a {page_w}x{page_h} page")
        out.append((x0, y0, x1 - x0, y1 - y0))
    return out


def _align(x, m=GEN_ALIGN):
    return max(m, int(round(float(x) / m) * m))


def gen_size(rect_w, rect_h, target_pixels=1024 * 1024, align=GEN_ALIGN, max_side=2048):
    """Taille de GENERATION d'une case: garde le ratio du rectangle final, vise
    ~target_pixels de surface, aligne sur `align`, plafonne le grand cote.

    On ne genere pas a la taille d'impression (une case A4 300dpi fait 2000+ px de
    haut, hors budget VRAM et hors distribution du modele): on genere a resolution
    de travail au BON RATIO, l'upscale se fait a l'export."""
    if rect_w <= 0 or rect_h <= 0:
        raise ValueError(f"invalid rect {rect_w}x{rect_h}")
    ar = float(rect_w) / float(rect_h)
    h = math.sqrt(float(target_pixels) / ar)
    w = ar * h
    big = max(w, h)
    if big > max_side:
        k = max_side / big
        w, h = w * k, h * k
    return _align(w, align), _align(h, align)


def page_size(preset_or_dict):
    """Resout un format de page: nom de PAGE_PRESETS ou dict deja complet."""
    if isinstance(preset_or_dict, str):
        p = PAGE_PRESETS.get(preset_or_dict)
        if p is None:
            raise ValueError(f"unknown page preset '{preset_or_dict}' "
                             f"(known: {', '.join(sorted(PAGE_PRESETS))})")
        preset_or_dict = p
    page = dict(DEFAULT_PAGE)
    page.update(preset_or_dict or {})
    return page


# ----------------------------------------------------------------------------
# Casting: @Name -> description + references Omni + LoRA
# ----------------------------------------------------------------------------
# @@ = un @ litteral. @Name = une entree de casting.
_AT = re.compile(r"@@|@([A-Za-z0-9_\-]+)")


def new_character(desc, refs=None, lora=None, negative="", kind="character"):
    """Fiche de casting. `lora` = 'fichier.safetensors:0.85' ou une liste.
    `kind` = 'character' (defaut) ou 'setting' (decor/lieu): les deux se
    substituent pareil dans les prompts, mais un decor n'est JAMAIS choisi par
    detail_prompt() comme sujet d'une passe de detail (un crop de visage refine
    avec 'a ruined castle' derive vers le chateau). Les fiches d'anciens
    project.json sans 'kind' sont lues comme 'character'."""
    if kind not in ("character", "setting"):
        raise ValueError(f"kind must be 'character' or 'setting', got {kind!r}")
    loras = [lora] if isinstance(lora, str) else list(lora or [])
    return {"desc": (desc or "").strip(), "refs": list(refs or []),
            "loras": [l for l in loras if l], "negative": (negative or "").strip(),
            "kind": kind}


def _casting_lookup(casting, name):
    """(cle canonique, fiche) pour un @Name: recherche exacte puis insensible a la
    casse (le scenariste tape @hero ou @Hero). (None, None) si absent. On renvoie la
    CLE et pas seulement la fiche: c'est elle qui sert a dedupliquer, sinon @hero et
    @Hero comptent comme deux personnages differents."""
    if name in casting:
        return name, casting[name]
    low = name.lower()
    for key, val in casting.items():
        if key.lower() == low:
            return key, val
    return None, None


def _casting_get(casting, name):
    """Fiche d'un @Name (voir _casting_lookup), ou None."""
    return _casting_lookup(casting, name)[1]


def resolve_casting(text, casting, max_refs=None):
    """Remplace les @Name par leur description et collecte refs / LoRA / negatifs.

    Renvoie un dict:
      prompt   : texte avec les @Name substitues (@@ -> @)
      refs     : chemins de reference, dans l'ordre d'apparition, dedupliques
      loras    : specs LoRA 'nom[:poids]', dedupliquees par fichier (1er poids gagne)
      negative : negatifs cumules des personnages cites
      used     : noms de casting effectivement resolus
      unknown  : @Name absents du casting (le nom nu reste dans le prompt)

    Un @Name inconnu n'est PAS laisse tel quel dans le prompt: '@Superhero' partirait
    tel quel dans l'encodeur de texte. On garde le nom nu et on remonte l'oubli."""
    casting = casting or {}
    refs, loras, negs, used, unknown = [], [], [], [], []
    seen_lora_files = set()

    def _sub(m):
        if m.group(0) == "@@":
            return "@"
        name = m.group(1)
        key, char = _casting_lookup(casting, name)
        if not char:
            if name not in unknown:
                unknown.append(name)
            return name
        if key not in used:
            used.append(key)
        for r in char.get("refs") or []:
            if r not in refs:
                refs.append(r)
        for spec in char.get("loras") or []:
            fname = str(spec).split(":", 1)[0].strip().lower()
            if fname and fname not in seen_lora_files:
                seen_lora_files.add(fname)
                loras.append(spec)
        neg = (char.get("negative") or "").strip()
        if neg and neg not in negs:
            negs.append(neg)
        return (char.get("desc") or "").strip() or name

    prompt = _AT.sub(_sub, text or "")
    # La substitution laisse parfois des doubles espaces / virgules orphelines.
    prompt = re.sub(r"\s{2,}", " ", prompt).strip()
    prompt = re.sub(r"\s+,", ",", prompt).strip(" ,")
    if max_refs is not None:
        refs = refs[:int(max_refs)]
    return {"prompt": prompt, "refs": refs, "loras": loras,
            "negative": ", ".join(negs), "used": used, "unknown": unknown}


# ----------------------------------------------------------------------------
# Modele de projet
# ----------------------------------------------------------------------------
def new_project(name, description="", page=None, style=None, casting=None):
    return {
        "schema": SCHEMA_VERSION,
        "name": name or "Untitled",
        "description": description or "",
        "page": page_size(page or DEFAULT_PAGE),
        "style": {"prompt_suffix": "", "negative": "", "loras": [],
                  **(style or {})},
        "casting": dict(casting or {}),
        "chapters": [],
    }


def _next_id(existing, prefix, width=2):
    n = 1
    taken = {e.get("id") for e in existing}
    while f"{prefix}{n:0{width}d}" in taken:
        n += 1
    return f"{prefix}{n:0{width}d}"


def add_chapter(project, name, synopsis=""):
    chapter = {"id": _next_id(project["chapters"], "ch"), "name": name or "Chapter",
               "synopsis": synopsis or "", "pages": []}
    project["chapters"].append(chapter)
    return chapter


def new_panel(pid, text=""):
    return {"id": pid, "text": text or "", "seed": -1, "status": "draft",
            "image": None, "refs": [], "loras": [], "notes": ""}


def add_page(project, chapter_id, layout="4-grid", texts=None):
    """Ajoute une planche a un chapitre. Cree autant de panneaux que le gabarit a
    de cases; `texts` (optionnel) pre-remplit les textes dans l'ordre de lecture.

    Plus de textes que de cases = ERREUR, pas une troncature silencieuse: le
    decoupage d'un scenariste ne doit jamais disparaitre sans un mot. L'appelant
    choisit un gabarit plus grand ou coupe la planche en deux."""
    chapter = find_chapter(project, chapter_id)
    cells = layout_cells(layout)
    if texts and len(texts) > len(cells):
        raise ValueError(
            f"{len(texts)} texts for layout '{layout}' ({len(cells)} cells): "
            f"pick a larger layout or split the page - texts are never dropped")
    page = {"id": _next_id(chapter["pages"], "p"), "layout": layout, "panels": []}
    for i in range(len(cells)):
        txt = texts[i] if texts and i < len(texts) else ""
        page["panels"].append(new_panel(f"pn{i + 1}", txt))
    chapter["pages"].append(page)
    return page


def set_layout(project, chapter_id, page_id, layout):
    """Change le gabarit d'une planche en gardant le travail deja fait: les panneaux
    existants sont conserves dans l'ordre, les cases en trop sont ajoutees vides.

    Renvoie (page, removed): en reduisant, les panneaux excedentaires sont RETIRES
    de la planche mais RENDUS a l'appelant (texte, seed, image comprises) - a lui
    de les re-injecter ailleurs, de les proposer a l'utilisateur ou de les jeter
    en connaissance de cause. Rien n'est detruit silencieusement."""
    page = find_page(project, chapter_id, page_id)
    n = len(layout_cells(layout))
    panels = page["panels"]
    while len(panels) < n:
        panels.append(new_panel(f"pn{len(panels) + 1}"))
    removed = []
    if len(panels) > n:
        removed = panels[n:]
        del panels[n:]
    page["layout"] = layout
    return page, removed


def find_chapter(project, chapter_id):
    for c in project["chapters"]:
        if c["id"] == chapter_id:
            return c
    raise KeyError(f"chapter '{chapter_id}' not found")


def find_page(project, chapter_id, page_id):
    for p in find_chapter(project, chapter_id)["pages"]:
        if p["id"] == page_id:
            return p
    raise KeyError(f"page '{page_id}' not found in chapter '{chapter_id}'")


def find_panel(project, chapter_id, page_id, panel_id):
    for pn in find_page(project, chapter_id, page_id)["panels"]:
        if pn["id"] == panel_id:
            return pn
    raise KeyError(f"panel '{panel_id}' not found in {chapter_id}/{page_id}")


def iter_panels(project):
    """(chapter, page, panel, index_de_case) sur tout le projet, dans l'ordre."""
    for chapter in project["chapters"]:
        for page in chapter["pages"]:
            for i, panel in enumerate(page["panels"]):
                yield chapter, page, panel, i


def panel_path(project_dir, chapter_id, page_id, panel_id, ext="png"):
    return os.path.join(project_dir, "panels", chapter_id, page_id, f"{panel_id}.{ext}")


def page_path(project_dir, chapter_id, page_id, ext="png"):
    return os.path.join(project_dir, "pages", chapter_id, f"{page_id}.{ext}")


# ----------------------------------------------------------------------------
# Ce qu'il faut envoyer au moteur pour UNE case
# ----------------------------------------------------------------------------
def resolve_panel(project, page, panel, index=None, target_pixels=1024 * 1024):
    """Tout ce dont l'appelant a besoin pour generer ce panneau: prompt resolu,
    negatif, refs Omni, LoRA, et la taille de generation au ratio de la case.

    Le style du projet est applique en SUFFIXE (apres le texte du panneau) et ses
    LoRA passent en dernier: une LoRA de personnage prime sur la LoRA de style si
    les deux designent le meme fichier."""
    cells = layout_cells(page["layout"])
    if index is None:
        index = page["panels"].index(panel)
    if index >= len(cells):
        raise ValueError(f"panel {panel['id']} has no cell in layout '{page['layout']}'")
    pg = page_size(project.get("page"))
    rects = panel_rects(cells, pg["width"], pg["height"], pg["margin"], pg["gutter"])
    rw, rh = rects[index][2], rects[index][3]

    res = resolve_casting(panel.get("text", ""), project.get("casting"))
    style = project.get("style") or {}
    parts = [res["prompt"], (style.get("prompt_suffix") or "").strip()]
    prompt = ", ".join(p for p in parts if p)
    negs = [res["negative"], (style.get("negative") or "").strip()]

    loras = list(res["loras"]) + list(panel.get("loras") or [])
    seen = set()
    merged = []
    for spec in loras + list(style.get("loras") or []):
        key = str(spec).split(":", 1)[0].strip().lower()
        if key and key not in seen:
            seen.add(key)
            merged.append(spec)

    refs = list(res["refs"])
    for r in panel.get("refs") or []:
        if r not in refs:
            refs.append(r)

    gw, gh = gen_size(rw, rh, target_pixels=target_pixels)
    return {"prompt": prompt,
            "negative": ", ".join(n for n in negs if n),
            "refs": refs, "loras": merged,
            "width": gw, "height": gh,
            "rect": rects[index], "seed": int(panel.get("seed", -1)),
            "unknown": res["unknown"]}


def detail_prompt(project, panel, subject=None):
    """Prompt LOCAL pour une passe de detail (visage / main) sur un CROP du panneau.

    Ne renvoie JAMAIS le texte de scene: envoyer le prompt global sur un fragment
    fait deriver le crop vers la scene entiere (bug corrige quatre fois dans ce
    repo -- voir l'historique du detailer). On renvoie la description du personnage
    cite en premier, ou `subject` si l'appelant sait mieux, ou une chaine vide -- un
    prompt vide est un resultat VALIDE et sur, pas un cas d'echec."""
    if subject:
        return subject.strip()
    text = panel.get("text") or ""
    m = _AT.search(text)
    while m:
        if m.group(0) != "@@":
            char = _casting_get(project.get("casting") or {}, m.group(1))
            # kind 'setting' saute: 'wide shot of @Castle, @Hero on the ramparts'
            # doit detailler Hero, pas renvoyer le chateau comme sujet de visage.
            if char and char.get("kind", "character") == "character":
                return (char.get("desc") or "").strip()
        m = _AT.search(text, m.end())
    return ""


# ----------------------------------------------------------------------------
# Composition de la planche
# ----------------------------------------------------------------------------
def _placeholder_font(px):
    """Police du label de placeholder, proportionnelle a la case. La bitmap PIL
    par defaut fait ~10 px: invisible sur une planche de 2048 px de large."""
    from PIL import ImageFont
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except Exception:
            continue
    return ImageFont.load_default()


def _placeholder(size, label, background="#ffffff"):
    img = Image.new("RGB", size, background)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, size[0] - 1, size[1] - 1], outline="#b0b0b0", width=max(2, size[0] // 200))
    d.line([0, 0, size[0] - 1, size[1] - 1], fill="#e0e0e0", width=max(1, size[0] // 300))
    d.line([0, size[1] - 1, size[0] - 1, 0], fill="#e0e0e0", width=max(1, size[0] // 300))
    d.text((size[0] // 2, size[1] // 2), label, fill="#808080", anchor="mm",
           font=_placeholder_font(max(14, min(size) // 10)))
    return img


def compose_page(project, page, images=None, fit="cover", placeholders=True):
    """Assemble une planche PIL a partir des images de ses panneaux.

    images : dict panel_id -> PIL.Image (prioritaire), sinon on charge panel['image'].
    fit    : 'cover' = remplit la case et recadre au centre (defaut, sans bande vide);
             'contain' = image entiere, fond visible autour.
    placeholders : dessine une case barree numerotee pour les panneaux non rendus."""
    pg = page_size(project.get("page"))
    cells = layout_cells(page["layout"])
    rects = panel_rects(cells, pg["width"], pg["height"], pg["margin"], pg["gutter"])
    sheet = Image.new("RGB", (pg["width"], pg["height"]), pg["background"])
    draw = ImageDraw.Draw(sheet)
    border = int(pg.get("border") or 0)

    for i, rect in enumerate(rects):
        if i >= len(page["panels"]):
            break
        panel = page["panels"][i]
        x, y, w, h = rect
        img = (images or {}).get(panel["id"])
        if img is None and panel.get("image") and os.path.isfile(panel["image"]):
            img = Image.open(panel["image"])
        if img is None:
            if not placeholders:
                continue
            img = _placeholder((w, h), f"{page['id']}.{panel['id']}", pg["background"])
        else:
            img = img.convert("RGB")
            if fit == "contain":
                canvas = Image.new("RGB", (w, h), pg["background"])
                scaled = ImageOps.contain(img, (w, h), Image.LANCZOS)
                canvas.paste(scaled, ((w - scaled.width) // 2, (h - scaled.height) // 2))
                img = canvas
            else:
                img = ImageOps.fit(img, (w, h), Image.LANCZOS)
        sheet.paste(img, (x, y))
        if border > 0:
            draw.rectangle([x, y, x + w - 1, y + h - 1],
                           outline=pg.get("border_color", "#000000"), width=border)
    return sheet


def export_pdf(images, path, dpi=300):
    """Multi-page PDF a partir d'une liste d'images de planches (ordre = pagination)."""
    imgs = [im.convert("RGB") for im in images if im is not None]
    if not imgs:
        raise ValueError("export_pdf: no page to export")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    imgs[0].save(path, "PDF", resolution=float(dpi), save_all=True,
                 append_images=imgs[1:])
    return path


# ----------------------------------------------------------------------------
# Persistance (project.json)
# ----------------------------------------------------------------------------
def project_json_path(project_dir):
    return os.path.join(project_dir, "project.json")


def save_project(project, project_dir):
    """Ecriture ATOMIQUE (tmp + os.replace): un crash en cours d'ecriture ne laisse
    jamais un project.json tronque -- c'est le seul endroit ou vit le scenario."""
    os.makedirs(project_dir, exist_ok=True)
    dst = project_json_path(project_dir)
    tmp = dst + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(project, f, indent=2, ensure_ascii=False)
    os.replace(tmp, dst)
    return dst


def load_project(project_dir):
    """Relit un projet et complete les cles absentes (tolerant aux fichiers ecrits
    par une version anterieure du schema)."""
    path = project_json_path(project_dir) if os.path.isdir(project_dir) else project_dir
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f) or {}
    if int(data.get("schema", 0)) > SCHEMA_VERSION:
        raise ValueError(f"project schema {data.get('schema')} is newer than this build "
                         f"(supports {SCHEMA_VERSION}); update crispz-studio")
    data.setdefault("schema", SCHEMA_VERSION)
    data.setdefault("name", "Untitled")
    data.setdefault("description", "")
    data["page"] = page_size(data.get("page"))
    style = data.get("style") or {}
    data["style"] = {"prompt_suffix": "", "negative": "", "loras": [], **style}
    data.setdefault("casting", {})
    data.setdefault("chapters", [])
    for chapter in data["chapters"]:
        chapter.setdefault("pages", [])
        for page in chapter["pages"]:
            page.setdefault("layout", "4-grid")
            for i, panel in enumerate(page.setdefault("panels", [])):
                base = new_panel(panel.get("id") or f"pn{i + 1}")
                base.update(panel)
                base["id"] = base["id"] or f"pn{i + 1}"
                page["panels"][i] = base
    return data
