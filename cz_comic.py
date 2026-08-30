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

# Role d'une planche dans le LIVRE. L'ordre de publication est trie par rang:
# les 'cover' ouvrent l'album, les 'back' le ferment - meme si des planches
# d'histoire sont ajoutees apres coup. 'title' = page de garde d'un chapitre
# (traitee comme l'histoire pour l'ordre, mais exclue de la numerotation).
# Les anciens project.json sans 'role' sont lus comme 'story'.
PAGE_ROLES = ("cover", "title", "story", "back")
_ROLE_RANK = {"cover": 0, "title": 1, "story": 1, "back": 2}


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


def add_page(project, chapter_id, layout="4-grid", texts=None, role="story"):
    """Ajoute une planche a un chapitre. Cree autant de panneaux que le gabarit a
    de cases; `texts` (optionnel) pre-remplit les textes dans l'ordre de lecture.
    `role` (PAGE_ROLES) place la planche dans le livre: cover en tete, back en
    queue, title/story dans l'ordre du document.

    Plus de textes que de cases = ERREUR, pas une troncature silencieuse: le
    decoupage d'un scenariste ne doit jamais disparaitre sans un mot. L'appelant
    choisit un gabarit plus grand ou coupe la planche en deux."""
    if role not in PAGE_ROLES:
        raise ValueError(f"role must be one of {PAGE_ROLES}, got {role!r}")
    chapter = find_chapter(project, chapter_id)
    cells = layout_cells(layout)
    if texts and len(texts) > len(cells):
        raise ValueError(
            f"{len(texts)} texts for layout '{layout}' ({len(cells)} cells): "
            f"pick a larger layout or split the page - texts are never dropped")
    page = {"id": _next_id(chapter["pages"], "p"), "layout": layout,
            "role": role, "panels": []}
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


# ----------------------------------------------------------------------------
# Lettrage: dialogues, bulles, cartouches, onomatopees
# ----------------------------------------------------------------------------
# Le texte n'est JAMAIS demande au modele (Z-Image invente des lettres:
# 'Mendian Station'), il est dessine vectoriellement APRES la composition:
# editable sans regenerer l'image, traduisible, net a l'impression.
DIALOGUE_KINDS = ("speech", "thought", "caption", "sfx")

# Formes de bulle: round = ellipse (classique), rounded = rectangle a coins
# arrondis (compact, lecture dense), angular = polygone a pans coupes (voix
# dure, mecanique, cri). Resolution: style de la replique > style.bubble du
# projet > 'round'.
BUBBLE_STYLES = ("round", "rounded", "angular")

# Polices essayees dans l'ordre (Windows puis fallbacks libres).
_BUBBLE_FONTS = ("comicbd.ttf", "comic.ttf", "segoeui.ttf", "arial.ttf",
                 "DejaVuSans.ttf")
_SFX_FONTS = ("impact.ttf", "arialbd.ttf", "comicbd.ttf", "DejaVuSans-Bold.ttf")


def add_dialogue(panel, text, speaker=None, kind="speech", anchor=None,
                 style=None):
    """Ajoute une replique a une case. `anchor` = (fx, fy) en fractions de la
    case, vers quoi pointe la queue de la bulle (defaut: bouche du locuteur
    detectee, sinon bas de la bulle). `style` = forme de bulle (BUBBLE_STYLES),
    None = style du projet."""
    if kind not in DIALOGUE_KINDS:
        raise ValueError(f"kind must be one of {DIALOGUE_KINDS}, got {kind!r}")
    if style is not None and style not in BUBBLE_STYLES:
        raise ValueError(f"style must be one of {BUBBLE_STYLES}, got {style!r}")
    text = (text or "").strip()
    if not text:
        raise ValueError("empty dialogue text")
    d = {"speaker": (speaker or "").strip(), "text": text, "kind": kind}
    if anchor:
        d["anchor"] = [float(anchor[0]), float(anchor[1])]
    if style:
        d["style"] = style
    panel.setdefault("dialogue", []).append(d)
    return d


def parse_dialogue(block):
    """Syntaxe scenariste -> liste de repliques, une par ligne:
         Kira: On y va.                 -> speech (speaker Kira)
         Kira (think): Trop tard.       -> thought
         Rook (angular): The case stays -> speech, bulle a pans coupes
         Kira (think, rounded): ...     -> thought, rectangle arrondi
         CAP: Trois heures plus tot.    -> caption (narration)
         SFX: KRAK                      -> onomatopee
       Modificateurs entre parentheses (cumulables, virgule): think/thought =
       pensee; round/rounded/angular = forme de bulle (BUBBLE_STYLES). Une
       parenthese inconnue reste dans le nom du locuteur. Une ligne sans ':'
       est une caption. Lignes vides ignorees."""
    out = []
    for line in (block or "").splitlines():
        line = line.strip()
        if not line:
            continue
        head, sep, text = line.partition(":")
        if not sep or not text.strip():
            out.append({"speaker": "", "text": line, "kind": "caption"})
            continue
        head, text = head.strip(), text.strip()
        low = head.lower()
        if low == "cap":
            out.append({"speaker": "", "text": text, "kind": "caption"})
        elif low == "sfx":
            out.append({"speaker": "", "text": text, "kind": "sfx"})
        else:
            kind, style = "speech", None
            m = re.match(r"^(.*?)\s*\(([^)]+)\)$", head)
            if m:
                known = True
                k2, s2 = kind, style
                for tok in m.group(2).split(","):
                    tok = tok.strip().lower()
                    if tok in ("think", "thought", "pense"):
                        k2 = "thought"
                    elif tok in BUBBLE_STYLES:
                        s2 = tok
                    else:
                        known = False
                        break
                if known:
                    head, kind, style = m.group(1).strip(), k2, s2
            d = {"speaker": head, "text": text, "kind": kind}
            if style:
                d["style"] = style
            out.append(d)
    return out


def _font(candidates, px):
    from PIL import ImageFont
    for name in candidates:
        try:
            return ImageFont.truetype(name, px)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap(draw, text, font, max_w):
    """Coupe le texte en lignes tenant dans max_w pixels (par mots)."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if cur and draw.textlength(cand, font=font) > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines or [""]


def _overlap_area(a, b):
    """Aire d'intersection de deux rects (x, y, w, h)."""
    ox = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    oy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0, ox) * max(0, oy)


def _face_zone(box, panel_rect, grow=0.15):
    """Bbox visage (x1,y1,x2,y2) -> rect interdit (x,y,w,h), elargi de `grow`
    et borne a la case."""
    x1, y1, x2, y2 = box
    gx, gy = (x2 - x1) * grow, (y2 - y1) * grow
    x1, y1, x2, y2 = x1 - gx, y1 - gy, x2 + gx, y2 + gy
    px, py, pw, ph = panel_rect
    x1, y1 = max(px, x1), max(py, y1)
    x2, y2 = min(px + pw, x2), min(py + ph, y2)
    return (int(x1), int(y1), max(0, int(x2 - x1)), max(0, int(y2 - y1)))


def _place_rect(panel_rect, bw, bh, forbidden, taken, prefer):
    """Position d'un rect bw x bh dans la case: balaye du haut vers le bas,
    ordre des colonnes selon `prefer` ('left' / 'right' / 'center').

    Priorite absolue: ZERO recouvrement des zones visage (`forbidden`) et des
    bulles deja posees (`taken`). Si aucune position propre n'existe, renvoie
    celle qui recouvre le MOINS de visage (dernier recours, jamais silencieux:
    le placement est signale clipped=True dans le retour du lettrage)."""
    x, y, w, h = panel_rect
    pad = 8
    cols = {"left": [x + pad, x + w - bw - pad, x + (w - bw) // 2],
            "right": [x + w - bw - pad, x + pad, x + (w - bw) // 2],
            "center": [x + (w - bw) // 2, x + pad, x + w - bw - pad]}[prefer]
    step = max(16, bh // 3)
    best, best_ov = None, None
    yy = y + pad
    while yy + bh <= y + h - pad:
        for xx in cols:
            xx = max(x + 2, min(int(xx), x + w - bw - 2))
            r = (xx, yy, bw, bh)
            if any(_overlap_area(r, t) > 0 for t in taken):
                continue
            ov = sum(_overlap_area(r, f) for f in forbidden)
            if ov == 0:
                return r, True
            if best is None or ov < best_ov:
                best, best_ov = r, ov
        yy += step
    return (best or (cols[0], y + pad, bw, bh)), False


def _pos_rect(d, panel_rect, bw, bh, forbidden):
    """Position MANUELLE d'une bulle: d['pos'] = coin haut-gauche en FRACTIONS
    de case, ecrit par Comic Studio quand l'utilisateur deplace la bulle.
    Prioritaire sur le placement automatique (_place_rect), clampee dans la
    case. clean=False si elle recouvre un visage: on ne la re-deplace PAS
    (l'utilisateur l'a posee la volontairement), on le signale seulement.
    Renvoie ((x, y, w, h), clean) ou None si pas de position manuelle."""
    pos = d.get("pos")
    if not pos:
        return None
    x, y, w, h = panel_rect
    bx = max(x + 2, min(x + int(float(pos[0]) * w), x + w - bw - 2))
    by = max(y + 2, min(y + int(float(pos[1]) * h), y + h - bh - 2))
    r = (bx, by, bw, bh)
    clean = all(_overlap_area(r, f) == 0 for f in forbidden)
    return r, clean


def _tail_tip(bubble_center, mouth, face_box):
    """Tail tip, picked from where the bubble sits relative to the face -
    always DESIGNATING the mouth without ever crossing or covering the face:

      - bubble BELOW the face (the common case now that bubbles avoid
        faces): tip just UNDER the chin, at the mouth's x - the tail rises
        straight toward the mouth. A side tip here seemed to point at the
        cheek/ear.
      - otherwise (bubble above or beside): AT MOUTH HEIGHT, just beside
        the face, on the bubble's side - the readable cheek convention.
        (v1 stopped the tail where the mouth->bubble segment left the bbox:
        with a bubble above it exited through the FOREHEAD - fixed.)"""
    mx, my = mouth
    cx, cy = bubble_center
    x1, y1, x2, y2 = face_box
    if cy > y2:
        margin_y = max(6, 0.10 * (y2 - y1))
        return (int(mx), int(y2 + margin_y))
    margin = max(6, 0.12 * (x2 - x1))
    tip_x = x1 - margin if cx < mx else x2 + margin
    return (int(tip_x), int(my))


def _cos(a, b):
    """Similarite cosinus de deux vecteurs (listes de floats)."""
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(x * x for x in b))
    return num / (da * db) if da and db else 0.0


def _match_speakers(speakers, faces, char_embeddings=None, threshold=0.2):
    """Apparie les locuteurs (noms lower, ordre de citation) aux visages d'une
    case -> {speaker: face}.

    1. RECONNAISSANCE d'abord: si char_embeddings fournit l'embedding du
       portrait de reference d'un locuteur et que les visages detectes portent
       le leur, appariement glouton par meilleure similarite cosinus (>= threshold).
    2. Les locuteurs restants prennent les visages restants dans le SENS DE
       LECTURE (1er locuteur cite = visage le plus a gauche) - l'heuristique
       v1, qui reste le fallback quand il n'y a pas de references."""
    result = {}
    remaining = list(range(len(faces)))
    if char_embeddings:
        pairs = []
        for s in speakers:
            emb = char_embeddings.get(s)
            if not emb:
                continue
            for j in remaining:
                fe = faces[j].get("embedding")
                if fe:
                    pairs.append((_cos(emb, fe), s, j))
        for score, s, j in sorted(pairs, key=lambda t: -t[0]):
            if score < threshold:
                break
            if s in result or j not in remaining:
                continue
            result[s] = faces[j]
            remaining.remove(j)
    for s in speakers:
        if s in result:
            continue
        if not remaining:
            break
        result[s] = faces[remaining.pop(0)]
    return result


def render_lettering(project, page, sheet, face_detector=None,
                     char_embeddings=None):
    """Dessine les dialogues sur la planche composee (in place) et renvoie la
    liste des placements [{panel, kind, rect, tip, clean}].

    Une replique peut porter d['pos'] = [fx, fy] (coin haut-gauche en fractions
    de case, ecrit par Comic Studio au drag): la bulle est alors posee LA,
    clampee dans la case, au lieu du placement automatique. d['anchor'] (deja
    en v1) pilote de la meme facon la POINTE de la queue. d['scale'] (0.4-3.0)
    est une HOMOTHETIE par replique: police, marges, queue et bulle grandissent
    ensemble, proportions conservees.

    Avec `face_detector` (callable image -> [{'box': (x1,y1,x2,y2),
    'mouth': (x,y)|None}], voir cz_face.detect_faces_full):
      - les bulles ne recouvrent JAMAIS un visage (zones interdites elargies;
        si la case est trop pleine, recouvrement minimal et clean=False);
      - la queue d'une bulle vise la BOUCHE de son locuteur: les locuteurs
        sont apparies aux visages dans le sens de lecture (1er locuteur =
        visage le plus a gauche), un `anchor` explicite gagne toujours;
      - un locuteur SANS visage (voix hors-champ, cri derriere, narrateur)
        recoit une bulle generique: queue vers le bord de case le plus proche.
    Sans detecteur: comportement v1 (empilage haut, alternance gauche/droite)."""
    pg = page_size(project.get("page"))
    cells = layout_cells(page["layout"])
    rects = panel_rects(cells, pg["width"], pg["height"], pg["margin"], pg["gutter"])
    draw = ImageDraw.Draw(sheet)
    placements = []

    for i, rect in enumerate(rects):
        if i >= len(page["panels"]):
            break
        panel = page["panels"][i]
        dialogue = panel.get("dialogue") or []
        if not dialogue:
            continue
        x, y, w, h = rect

        # --- visages de la case (coordonnees planche) ---
        faces = []
        if face_detector is not None:
            try:
                for f in face_detector(sheet.crop((x, y, x + w, y + h))) or []:
                    bx1, by1, bx2, by2 = f["box"]
                    faces.append({
                        "box": (x + bx1, y + by1, x + bx2, y + by2),
                        "mouth": ((x + f["mouth"][0], y + f["mouth"][1])
                                  if f.get("mouth") else None),
                        "embedding": f.get("embedding")})
            except Exception:
                faces = []
        faces.sort(key=lambda f: f["box"][0])            # sens de lecture
        forbidden = [_face_zone(f["box"], rect) for f in faces]

        # locuteur -> visage: reconnaissance par embeddings (portraits de
        # reference du casting) puis fallback sens de lecture
        speakers = []
        for d in dialogue:
            s = (d.get("speaker") or "").strip().lower()
            if d.get("kind") in ("speech", "thought") and s and s not in speakers:
                speakers.append(s)
        face_of = _match_speakers(speakers, faces, char_embeddings)

        fpx = max(16, min(44, h // 22))
        font = _font(_BUBBLE_FONTS, fpx)
        pad = max(8, fpx // 2)
        taken = []
        side = 0

        for d in dialogue:
            kind = d.get("kind", "speech")
            text = d.get("text") or ""
            # Homothetie par replique (Comic Studio): d['scale'] multiplie la
            # police -> bulle ET texte grandissent ensemble, memes proportions.
            try:
                scale = float(d.get("scale") or 1.0)
            except (TypeError, ValueError):
                scale = 1.0
            scale = max(0.4, min(3.0, scale))
            fpx_d = fpx if scale == 1.0 else max(10, int(round(fpx * scale)))
            font_d = font if scale == 1.0 else _font(_BUBBLE_FONTS, fpx_d)
            pad_d = max(8, fpx_d // 2)
            if kind == "sfx":
                # auto-fit: un cri long ou un TITRE de couverture ne doit pas
                # deborder la case -> la police retrecit jusqu'a tenir en
                # largeur (une echelle manuelle desserre/serre ce plafond)
                size = int(max(fpx * 2, h // 8) * scale)
                sfx_font = _font(_SFX_FONTS, size)
                bb = draw.textbbox((0, 0), text, font=sfx_font, stroke_width=4)
                while size > 10 and bb[2] - bb[0] > int(w * 0.92 * scale):
                    size = int(size * 0.85)
                    sfx_font = _font(_SFX_FONTS, size)
                    bb = draw.textbbox((0, 0), text, font=sfx_font, stroke_width=4)
                bw_, bh_ = bb[2] - bb[0], bb[3] - bb[1]
                (rx, ry, _, _), clean = _pos_rect(d, rect, bw_, bh_, forbidden) \
                    or _place_rect(rect, bw_, bh_, forbidden, taken, "center")
                draw.text((rx, ry), text, font=sfx_font, fill="#ffffff",
                          stroke_width=max(3, fpx_d // 5), stroke_fill="#000000")
                taken.append((rx, ry, bw_, bh_))
                placements.append({"panel": panel["id"], "kind": kind,
                                   "rect": (rx, ry, bw_, bh_), "tip": None,
                                   "clean": clean})
                continue

            # la largeur de coupe suit l'echelle (vraie homothetie: la bulle
            # garde ses proportions), plafonnee a la case
            max_text_w = min(int(w * 0.92),
                             int(w * (0.86 if kind == "caption" else 0.58)
                                 * scale))
            lines = _wrap(draw, text, font_d, max_text_w)
            line_h = fpx_d + 4
            text_w = max(int(draw.textlength(l, font=font_d)) for l in lines)
            text_h = line_h * len(lines)

            if kind == "caption":
                bw_, bh_ = text_w + pad_d * 2, text_h + pad_d * 2
                (bx0, by0, _, _), clean = _pos_rect(d, rect, bw_, bh_, forbidden) \
                    or _place_rect(rect, bw_, bh_, forbidden, taken, "left")
                draw.rectangle([bx0, by0, bx0 + bw_, by0 + bh_],
                               fill="#fdf6d8", outline="#000000", width=3)
                ty = by0 + pad_d
                for l in lines:
                    draw.text((bx0 + pad_d, ty), l, font=font_d, fill="#000000")
                    ty += line_h
                taken.append((bx0, by0, bw_, bh_))
                placements.append({"panel": panel["id"], "kind": kind,
                                   "rect": (bx0, by0, bw_, bh_), "tip": None,
                                   "clean": clean})
                continue

            # --- speech / thought ---
            bstyle = d.get("style") or (project.get("style") or {}).get("bubble") \
                or "round"
            if bstyle not in BUBBLE_STYLES:
                bstyle = "round"
            if bstyle == "round":     # ellipse: le texte tient dans l'inscrite
                bw_ = int((text_w + pad_d * 2) * 1.25)
                bh_ = int((text_h + pad_d * 2) * 1.45)
            else:                     # rectangle arrondi / pans coupes: compact
                bw_ = text_w + pad_d * 3
                bh_ = text_h + pad_d * 3
            prefer = "left" if side == 0 else "right"
            spk = (d.get("speaker") or "").strip().lower()
            fc = face_of.get(spk)
            if fc:  # pres du locuteur: colonne du cote de son visage
                prefer = "left" if (fc["box"][0] + fc["box"][2]) / 2 < x + w / 2 \
                    else "right"
            (bx0, by0, _, _), clean = _pos_rect(d, rect, bw_, bh_, forbidden) \
                or _place_rect(rect, bw_, bh_, forbidden, taken, prefer)
            cx, cy = bx0 + bw_ // 2, by0 + bh_ // 2

            # pointe de la queue: anchor explicite > bouche du locuteur > bord
            if d.get("anchor"):
                ax, ay = d["anchor"]
                tip = (max(x + 2, min(x + int(ax * w), x + w - 2)),
                       max(y + 2, min(y + int(ay * h), y + h - 2)))
            elif fc:
                bx1, by1, bx2, by2 = fc["box"]
                # sans keypoints: la bouche est ~au 4/5 de la hauteur du visage
                mouth = fc.get("mouth") or ((bx1 + bx2) / 2,
                                            by1 + 0.82 * (by2 - by1))
                tip = _tail_tip((cx, cy), mouth, fc["box"])
                tip = (max(x + 2, min(tip[0], x + w - 2)),
                       max(y + 2, min(tip[1], y + h - 2)))
            else:
                # No face info (no detector, or unmatched speaker): aim the
                # tail BELOW the bubble, slightly toward the panel centre -
                # where characters are drawn in the vast majority of panels.
                # (v1 aimed at the nearest vertical edge: without a face
                # detector EVERY tail seemed to point at nothing.)
                tx = cx + int((x + w / 2 - cx) * 0.35)
                tip = (max(x + 2, min(tx, x + w - 2)),
                       min(y + h - 4, by0 + bh_ + int(0.22 * h)))

            # base de la queue: bord de la bulle COTE pointe (bas si la pointe
            # est dessous, haut si elle est au-dessus de la bulle)
            tail_up = tip[1] < by0
            base_y = by0 + int(bh_ * (0.18 if tail_up else 0.82))
            base_out = base_y + (3 if tail_up else -3)
            if kind == "speech":
                draw.polygon([(cx - fpx_d // 2, base_y),
                              (cx + fpx_d // 2, base_y),
                              tip], fill="#ffffff", outline="#000000")
            if bstyle == "rounded":
                draw.rounded_rectangle([bx0, by0, bx0 + bw_, by0 + bh_],
                                       radius=max(8, min(bh_ // 3, fpx_d)),
                                       fill="#ffffff", outline="#000000", width=3)
            elif bstyle == "angular":
                c = max(6, min(bw_, bh_) // 5)
                pts = [(bx0 + c, by0), (bx0 + bw_ - c, by0),
                       (bx0 + bw_, by0 + c), (bx0 + bw_, by0 + bh_ - c),
                       (bx0 + bw_ - c, by0 + bh_), (bx0 + c, by0 + bh_),
                       (bx0, by0 + bh_ - c), (bx0, by0 + c)]
                draw.polygon(pts, fill="#ffffff")
                draw.line(pts + [pts[0]], fill="#000000", width=3, joint="curve")
            else:
                draw.ellipse([bx0, by0, bx0 + bw_, by0 + bh_],
                             fill="#ffffff", outline="#000000", width=3)
            if kind == "speech":
                draw.polygon([(cx - fpx_d // 2 + 3, base_out),
                              (cx + fpx_d // 2 - 3, base_out),
                              (tip[0], tip[1] + (4 if tail_up else -4))],
                             fill="#ffffff")
            else:
                # Thought circles trail from the bubble EDGE facing the tip
                # (exit point of the centre->tip ray from the bubble rect).
                # v1 always started from the bubble BOTTOM: with a tip above
                # (mouth above, bubble below the face) the segment crossed
                # the bubble and the circles landed on the text.
                dx, dy = tip[0] - cx, tip[1] - cy
                t = min((bw_ / 2) / abs(dx) if dx else float("inf"),
                        (bh_ / 2) / abs(dy) if dy else float("inf"))
                t = 1.0 if t == float("inf") else min(t, 1.0)
                ex, ey = cx + dx * t, cy + dy * t
                for k, r in ((0.35, max(3, fpx_d // 3)),
                             (0.65, max(2, fpx_d // 5))):
                    px_ = int(ex + (tip[0] - ex) * k)
                    py_ = int(ey + (tip[1] - ey) * k)
                    draw.ellipse([px_ - r, py_ - r, px_ + r, py_ + r],
                                 fill="#ffffff", outline="#000000", width=2)
            ty = by0 + (bh_ - text_h) // 2
            for l in lines:
                lw = draw.textlength(l, font=font)
                draw.text((bx0 + (bw_ - lw) // 2, ty), l, font=font,
                          fill="#000000")
                ty += line_h
            taken.append((bx0, by0, bw_, bh_))
            placements.append({"panel": panel["id"], "kind": kind,
                               "rect": (bx0, by0, bw_, bh_), "tip": tip,
                               "clean": clean})
            side = 1 - side
    return placements


# ----------------------------------------------------------------------------
# Character sheets & exports
# ----------------------------------------------------------------------------
def sheet_prompt(char, style=None):
    """(prompt, negative) pour generer la planche de reference d'une fiche de
    casting: un portrait canonique (seed fixe cote appelant) qui sert ensuite
    de ref Omni. Pour un decor (kind 'setting'): un plan d'ensemble vide."""
    desc = (char.get("desc") or "").strip()
    style = style or {}
    if char.get("kind", "character") == "setting":
        base = f"{desc}, wide establishing shot, empty scene, no people"
    else:
        base = (f"character reference sheet of {desc}, front view portrait and "
                f"full body, neutral grey background, consistent design")
    suffix = (style.get("prompt_suffix") or "").strip()
    prompt = ", ".join(p for p in (base, suffix) if p)
    negs = [char.get("negative") or "", style.get("negative") or ""]
    return prompt, ", ".join(n for n in negs if n)


def export_cbz(pages, path):
    """CBZ (format standard des liseuses BD): zip d'images numerotees.
    `pages` = images PIL ou chemins de fichiers, dans l'ordre de pagination."""
    import io
    import zipfile
    if not pages:
        raise ValueError("export_cbz: no page to export")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for i, p in enumerate(pages):
            name = f"{i + 1:03d}.png"
            if isinstance(p, str):
                z.write(p, name)
            else:
                buf = io.BytesIO()
                p.convert("RGB").save(buf, "PNG")
                z.writestr(name, buf.getvalue())
    return path


# ----------------------------------------------------------------------------
# Orchestration de rendu (le moteur est INJECTE: testable sans GPU)
# ----------------------------------------------------------------------------
def render_project(project, project_dir, engine, only=None, force=False,
                   progress=None):
    """Genere les cases du projet via `engine(spec) -> PIL.Image`.

    spec = resolve_panel() + {'chapter','page','panel'} (les ids). Une case qui
    a deja une image est sautee sauf `force`. `only` filtre par id complet
    'ch01.p02.pn3' ou par prefixe ('ch01', 'ch01.p02'). L'image est sauvee via
    panel_path() et le project.json est mis a jour apres CHAQUE case (un crash
    au milieu ne perd rien). Renvoie la liste des ids rendus."""
    only = set(only or [])

    def _selected(cid, pid, pnid):
        if not only:
            return True
        full = f"{cid}.{pid}.{pnid}"
        return any(full == o or full.startswith(o + ".") for o in only)

    rendered = []
    for chapter, page, panel, i in iter_panels(project):
        pnid = f"{chapter['id']}.{page['id']}.{panel['id']}"
        if not _selected(chapter["id"], page["id"], panel["id"]):
            continue
        if panel.get("image") and os.path.isfile(panel["image"]) and not force:
            continue
        spec = resolve_panel(project, page, panel, index=i)
        spec.update({"chapter": chapter["id"], "page": page["id"],
                     "panel": panel["id"]})
        if progress:
            progress(pnid, spec)
        img = engine(spec)
        if img is None:
            continue
        dst = panel_path(project_dir, chapter["id"], page["id"], panel["id"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        img.save(dst)
        panel["image"] = dst
        panel["status"] = "rendered"
        save_project(project, project_dir)
        rendered.append(pnid)
    return rendered


def set_page_role(project, chapter_id, page_id, role):
    """Change le role d'une planche dans le livre (PAGE_ROLES)."""
    if role not in PAGE_ROLES:
        raise ValueError(f"role must be one of {PAGE_ROLES}, got {role!r}")
    page = find_page(project, chapter_id, page_id)
    page["role"] = role
    return page


def book_order(project):
    """(chapter, page) dans l'ORDRE DE PUBLICATION: les 'cover' d'abord, puis
    title/story chapitre par chapitre dans l'ordre du document, les 'back' en
    dernier - meme si des planches ont ete ajoutees apres le dos. Tri STABLE:
    a rang egal, l'ordre du document est conserve."""
    flat = [(ch, pg) for ch in project["chapters"] for pg in ch["pages"]]
    return sorted(flat, key=lambda t: _ROLE_RANK.get(t[1].get("role", "story"), 1))


def _draw_page_number(sheet, pg, number):
    """Folio bas-centre, dans la marge (jamais sur les cases)."""
    draw = ImageDraw.Draw(sheet)
    margin = int(pg.get("margin") or 0)
    fpx = max(14, min(36, margin - 8)) if margin >= 24 else 0
    if not fpx:
        return                                  # pas de marge = pas de folio
    font = _font(_BUBBLE_FONTS, fpx)
    text = str(number)
    tw = draw.textlength(text, font=font)
    try:                                        # encre selon la luminance du fond
        from PIL import ImageColor
        r, g, b = ImageColor.getrgb(pg.get("background", "#ffffff"))[:3]
        ink = "#000000" if (0.299 * r + 0.587 * g + 0.114 * b) > 128 else "#e8e8e8"
    except Exception:
        ink = "#000000"
    draw.text(((pg["width"] - tw) // 2, pg["height"] - margin + (margin - fpx) // 2),
              text, font=font, fill=ink)


def compose_book(project, project_dir, letter=True, fit="cover",
                 face_detector=None, char_embeddings=None, numbers=None):
    """Compose et sauve TOUT le livre dans l'ordre de publication (book_order):
    couvertures, chapitres, dos. Renvoie la liste des chemins, prete pour
    export_pdf / export_cbz.

    numbers: None = suit project['page']['page_numbers'] (defaut False, pour ne
    pas alterer les albums existants); True/False force. Seules les planches
    'story' sont foliotees (1, 2, ...) - couvertures, pages de garde et dos
    n'ont jamais de numero."""
    pg_conf = page_size(project.get("page"))
    if numbers is None:
        numbers = bool(pg_conf.get("page_numbers", False))
    paths, folio = [], 0
    for chapter, page in book_order(project):
        sheet = compose_page(project, page, fit=fit)
        if letter:
            render_lettering(project, page, sheet, face_detector=face_detector,
                             char_embeddings=char_embeddings)
        if page.get("role", "story") == "story":
            folio += 1
            if numbers:
                _draw_page_number(sheet, pg_conf, folio)
        dst = page_path(project_dir, chapter["id"], page["id"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        sheet.save(dst)
        paths.append(dst)
    return paths


def compose_chapter(project, project_dir, chapter_id, letter=True, fit="cover",
                    face_detector=None, char_embeddings=None):
    """Compose et sauve toutes les planches d'un chapitre (+ lettrage), renvoie
    la liste des chemins dans l'ordre de pagination."""
    chapter = find_chapter(project, chapter_id)
    paths = []
    for page in chapter["pages"]:
        sheet = compose_page(project, page, fit=fit)
        if letter:
            render_lettering(project, page, sheet, face_detector=face_detector,
                             char_embeddings=char_embeddings)
        dst = page_path(project_dir, chapter_id, page["id"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        sheet.save(dst)
        paths.append(dst)
    return paths
