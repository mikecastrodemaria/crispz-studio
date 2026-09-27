"""crispz-studio - comic: the comic project model, the plate layouts, the cast.

A PURELY geometric and documentary layer: no torch / cz_pipeline import, so it is
importable and testable without a GPU (the tests run in <1s). The engine (txt2img,
omni, inpaint) is called by the caller -- this module tells it WHAT to generate, at
WHAT size, with WHICH references, and recomposes the plate afterwards.

It holds:
  - LAYOUTS / PAGE_PRESETS : the plate templates (fractions) and the page formats
  - panel_rects / gen_size : the panels' geometry -> pixels, and the generation size
  - casting : the resolution of the @Name -> a description + Omni refs + a LoRA
  - the project : new_project / load_project / save_project / add_chapter / add_page
  - compose_page / export_pdf : the assembly of the final plate

An important principle (a reminder of the bug fixed several times here): a FRAGMENT never
receives the scene's prompt. The detail pass on a crop goes through detail_prompt(),
which returns a LOCAL description, never the panel's text. See detail_prompt().

"""

import os
import re
import sys
import json
import math
import random

from PIL import Image, ImageDraw, ImageOps

from prompt_variants import expand_variants, has_variants

SCHEMA_VERSION = 1

# A tolerance on the layout fractions (an edge at 0.5 must be recognised as
# interior, an edge at 1.0 as exterior, despite the floats).
_EPS = 1e-6

# The alignment of the generation dimensions. 32 and not 16: the Z-Image transformer
# patchifies the VAE latent by 2 (see cz_pipeline.round_to_multiple).
GEN_ALIGN = 32


# ----------------------------------------------------------------------------
# Templates
# ----------------------------------------------------------------------------
# A panel = (x, y, w, h) in FRACTIONS of the usable area (the page minus the margins).
# The list's order = the panels' reading order.
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

# The common plate formats. dpi serves the export (PDF) and nothing else.
PAGE_PRESETS = {
    # the traditional print formats (300 dpi): a margin of ~2 cm (236 px), a gutter of
    # 5 mm (59 px) - the customary print values, changeable afterwards
    "Franco-Belge 24x32 cm":   {"width": 2835, "height": 3780, "dpi": 300, "margin": 236, "gutter": 59},
    "A4 300dpi":               {"width": 2480, "height": 3508, "dpi": 300, "margin": 236, "gutter": 59},
    "US comic 17x26 cm":       {"width": 1988, "height": 3075, "dpi": 300, "margin": 200, "gutter": 59},
    "Manga 13x18 cm":          {"width": 1535, "height": 2126, "dpi": 300, "margin": 150, "gutter": 47},
    "Graphic novel 17x24 cm":  {"width": 2008, "height": 2835, "dpi": 300, "margin": 200, "gutter": 59},
    "Square album 21x21 cm":   {"width": 2480, "height": 2480, "dpi": 300, "margin": 236, "gutter": 59},
    "Landscape 29.7x21 cm":    {"width": 3508, "height": 2480, "dpi": 300, "margin": 236, "gutter": 59},
    # digital
    "Web":                     {"width": 1280, "height": 1980, "dpi": 96, "margin": 96, "gutter": 48},
    "Webtoon":                 {"width": 800, "height": 1280, "dpi": 96, "margin": 40, "gutter": 80},
}

# One line of explanation per format, for the wizard (never in project.json)
PAGE_NOTES = {
    "Franco-Belge 24x32 cm": "hardcover album, colour, 48-64 pages",
    "A4 300dpi": "Franco-Belge A4 21x29.7 cm, print",
    "US comic 17x26 cm": "stapled comic book, 22-32 pages per issue",
    "Manga 13x18 cm": "pocket size, black & white, reads right to left",
    "Graphic novel 17x24 cm": "thicker book, 100+ pages, soft or hard cover",
    "Square album 21x21 cm": "children's picture book",
    "Landscape 29.7x21 cm": "landscape album",
    "Web": "screen only, fastest",
    "Webtoon": "tall pages for a vertical phone scroll",
}

DEFAULT_PAGE = {
    "width": 2480, "height": 3508, "dpi": 300,
    "margin": 96,          # the white around the usable area (px)
    "gutter": 48,          # the gutter BETWEEN two panels (px)
    "background": "#ffffff",
    "border": 0,           # a black frame around every panel (px, 0 = none)
    "border_color": "#000000",
}

PANEL_STATUS = ("draft", "locked")

# A plate's role in the BOOK. The publication order is sorted by rank:
# the 'cover' ones open the album, the 'back' ones close it - even when story
# plates are added afterwards. 'title' = a chapter's flyleaf (treated like the
# story for the order, but excluded from the numbering).
# The old project.json with no 'role' are read as 'story'.
PAGE_ROLES = ("cover", "title", "story", "back")
_ROLE_RANK = {"cover": 0, "title": 1, "story": 1, "back": 2}


def layout_names():
    return sorted(LAYOUTS)


def layout_cells(name):
    """The panels of a template. Raises ValueError when the name is unknown."""
    cells = LAYOUTS.get(name)
    if cells is None:
        raise ValueError(f"unknown layout '{name}' (known: {', '.join(layout_names())})")
    return list(cells)


def validate_cells(cells):
    """Checks that a template fits in [0,1] and that its panels do not overlap.
    Returns the list of problems (empty = a sound template)."""
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
# Geometry
# ----------------------------------------------------------------------------
def panel_rects(cells, page_w, page_h, margin=0, gutter=0):
    """Fractions -> pixel rectangles (x, y, w, h) on the plate.

    Every panel is pulled in by gutter/2 on its INTERIOR edges only: so two neighbouring
    panels are separated by exactly `gutter`, and the edge panels touch the margin
    exactly (no stray half-gutter at the edge of the page)."""
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
    """A panel's GENERATION size: it keeps the final rectangle's ratio, aims at
    ~target_pixels of area, aligns on `align`, caps the long side.

    We do not generate at the printing size (an A4 300dpi panel is 2000+ px tall, outside
    the VRAM budget and outside the model's distribution): we generate at a working
    resolution at the RIGHT RATIO, and the upscale happens at export time."""
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
    """Resolves a page format: a PAGE_PRESETS name or an already complete dict."""
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
# @@ = a literal @. @Name = a cast entry.
_AT = re.compile(r"@@|@([A-Za-z0-9_\-]+)")


def new_character(desc, refs=None, lora=None, negative="", kind="character"):
    """A cast sheet. `lora` = 'file.safetensors:0.85' or a list.
    `kind` = 'character' (the default) or 'setting' (scenery/a place): both substitute
    the same way in the prompts, but a setting is NEVER chosen by detail_prompt() as
    the subject of a detail pass (a face crop refined with 'a ruined castle' drifts
    towards the castle). The sheets of old project.json with no 'kind' are read as
    'character'."""
    if kind not in ("character", "setting"):
        raise ValueError(f"kind must be 'character' or 'setting', got {kind!r}")
    loras = [lora] if isinstance(lora, str) else list(lora or [])
    return {"desc": (desc or "").strip(), "refs": list(refs or []),
            "loras": [l for l in loras if l], "negative": (negative or "").strip(),
            "kind": kind}


def _casting_lookup(casting, name):
    """(the canonical key, the sheet) for a @Name: an exact search then a
    case-insensitive one (the scriptwriter types @hero or @Hero). (None, None) when
    absent. We return the KEY and not only the sheet: it is the key that serves to
    deduplicate, otherwise @hero and @Hero count as two different characters."""
    if name in casting:
        return name, casting[name]
    low = name.lower()
    for key, val in casting.items():
        if key.lower() == low:
            return key, val
    return None, None


def _casting_get(casting, name):
    """The sheet of a @Name (see _casting_lookup), or None."""
    return _casting_lookup(casting, name)[1]


def resolve_casting(text, casting, max_refs=None):
    """Replaces the @Name with their description and collects the refs / LoRAs / negatives.

    Returns a dict:
      prompt   : the text with the @Name substituted (@@ -> @)
      refs     : the reference paths, in order of appearance, deduplicated
      loras    : the LoRA specs 'name[:weight]', deduplicated by file (the 1st weight wins)
      negative : the accumulated negatives of the characters cited
      used     : the cast names actually resolved
      unknown  : the @Name absent from the cast (the bare name stays in the prompt)

    A @Name that is unknown is NOT left as it is in the prompt: '@Superhero' would go to
    the text encoder as it is. We keep the bare name and report the omission."""
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
    # The substitution sometimes leaves double spaces / orphan commas.
    prompt = re.sub(r"\s{2,}", " ", prompt).strip()
    prompt = re.sub(r"\s+,", ",", prompt).strip(" ,")
    if max_refs is not None:
        refs = refs[:int(max_refs)]
    return {"prompt": prompt, "refs": refs, "loras": loras,
            "negative": ", ".join(negs), "used": used, "unknown": unknown}


# ----------------------------------------------------------------------------
# The project model
# ----------------------------------------------------------------------------
def new_project(name, description="", page=None, style=None, casting=None):
    return {
        "schema": SCHEMA_VERSION,
        "name": name or "Untitled",
        "description": description or "",
        "page": page_size(page or DEFAULT_PAGE),
        "style": {"prompt_suffix": "", "negative": "", "loras": [], "mood": "",
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


def add_chapter(project, name, synopsis="", mood=""):
    """`mood` = THE CHAPTER's visual atmosphere (palette, light, weather...),
    which replaces the style's global mood (style['mood']) for its plates.
    Empty = the global mood applies."""
    chapter = {"id": _next_id(project["chapters"], "ch"), "name": name or "Chapter",
               "synopsis": synopsis or "", "pages": [], "mood": (mood or "").strip()}
    project["chapters"].append(chapter)
    return chapter


def chapter_of_page(project, page):
    """The chapter that holds this plate (None when there is none). `page` =
    the plate's dict (object identity: the plate ids p01, p02... repeat from one
    chapter to the next) or its id when it is unique."""
    chapters = project.get("chapters") or []
    if isinstance(page, dict):
        for ch in chapters:
            if any(pg is page for pg in ch.get("pages") or []):
                return ch
        page = page.get("id")
    hits = [ch for ch in chapters
            if any(pg.get("id") == page for pg in ch.get("pages") or [])]
    return hits[0] if len(hits) == 1 else None


def effective_mood(project, page):
    """The mood applied to a plate: the chapter's when it is filled in,
    otherwise the style's global mood (style['mood']), otherwise nothing."""
    ch = chapter_of_page(project, page) if page else None
    mood = (ch or {}).get("mood") or ""
    if not str(mood).strip():
        mood = (project.get("style") or {}).get("mood") or ""
    return str(mood).strip()


def new_panel(pid, text=""):
    return {"id": pid, "text": text or "", "seed": -1, "status": "draft",
            "image": None, "refs": [], "loras": [], "notes": ""}


def add_page(project, chapter_id, layout="4-grid", texts=None, role="story"):
    """Adds a plate to a chapter. It creates as many panels as the template has
    panels; `texts` (optional) pre-fills the texts in reading order.
    `role` (PAGE_ROLES) places the plate in the book: cover at the head, back at the
    tail, title/story in the document's order.

    More texts than panels = an ERROR, not a silent truncation: a scriptwriter's
    breakdown must never disappear without a word. The caller picks a bigger template
    or cuts the plate in two."""
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
    """Changes a plate's template while keeping the work already done: the existing
    panels are kept in order, and the extra panels are added empty.

    Returns (page, removed): on a reduction, the surplus panels are REMOVED from the
    plate but RETURNED to the caller (text, seed, image included) - it is up to it to
    re-inject them elsewhere, to offer them to the user or to throw them away
    knowingly. Nothing is destroyed silently."""
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
    """(chapter, page, panel, panel_index) over the whole project, in order."""
    for chapter in project["chapters"]:
        for page in chapter["pages"]:
            for i, panel in enumerate(page["panels"]):
                yield chapter, page, panel, i


def panel_path(project_dir, chapter_id, page_id, panel_id, ext="png"):
    return os.path.join(project_dir, "panels", chapter_id, page_id, f"{panel_id}.{ext}")


def page_path(project_dir, chapter_id, page_id, ext="png"):
    return os.path.join(project_dir, "pages", chapter_id, f"{page_id}.{ext}")


# ----------------------------------------------------------------------------
# The {a|b|c} variants (prompt_variants, the syntax shared by the whole family)
# ----------------------------------------------------------------------------
def _expand_text(text, seed):
    """Expands the {a|b|c} groups of ONE field with its own random.Random(seed):
    the same seed + the same text = the same choice, whatever the caller (the panel's
    render, a variation, a detail pass). A text with no group is returned as it is,
    with no draw at all."""
    if not text or not has_variants(text):
        return text
    out = expand_variants(text, random.Random(int(seed)) if int(seed) >= 0
                          else random.Random())
    print(f"[Variants] {text} -> {out}", file=sys.stderr, flush=True)
    return out


def _panel_variant_texts(project, page, panel):
    """Every text that makes up this panel's prompt, raw: the panel's text,
    the style, the atmosphere, and the description / negative of the sheets cited."""
    casting = project.get("casting") or {}
    style = project.get("style") or {}
    texts = [panel.get("text") or "", style.get("prompt_suffix") or "",
             style.get("negative") or "", effective_mood(project, page) or ""]
    for name in resolve_casting(panel.get("text") or "", casting)["used"]:
        char = casting.get(name) or {}
        texts += [char.get("desc") or "", char.get("negative") or ""]
    return texts


def panel_seed(project, page, panel):
    """The panel's render seed. A panel one of whose texts uses {a|b|c} and whose seed
    is -1 receives a CONCRETE seed, written into panel['seed'] (the caller saves the
    project as usual): the options drawn must survive a new render, a variation and the
    detail pass. With no group, nothing changes: -1 stays -1 (the engine draws the seed,
    as before)."""
    seed = int(panel.get("seed", -1))
    if seed < 0 and any(has_variants(t)
                        for t in _panel_variant_texts(project, page, panel)):
        seed = random.randint(0, 2**31 - 1)
        panel["seed"] = seed
    return seed


def _expanded_casting(casting, seed):
    """A copy of the cast whose desc / negative have their groups expanded (seed)."""
    out = {}
    for name, char in (casting or {}).items():
        c = dict(char)
        for key in ("desc", "negative"):
            if c.get(key):
                c[key] = _expand_text(c[key], seed)
        out[name] = c
    return out


# ----------------------------------------------------------------------------
# What has to be sent to the engine for ONE panel
# ----------------------------------------------------------------------------
def resolve_panel(project, page, panel, index=None, target_pixels=1024 * 1024):
    """Everything the caller needs to generate this panel: the resolved prompt, the
    negative, the Omni refs, the LoRAs, and the generation size at the panel's ratio.

    The project's style is applied as a SUFFIX (after the panel's text) and its LoRAs go
    last: a character LoRA wins over the style LoRA when both name the same file."""
    cells = layout_cells(page["layout"])
    if index is None:
        index = page["panels"].index(panel)
    if index >= len(cells):
        raise ValueError(f"panel {panel['id']} has no cell in layout '{page['layout']}'")
    pg = page_size(project.get("page"))
    rects = panel_rects(cells, pg["width"], pg["height"], pg["margin"], pg["gutter"])
    rw, rh = rects[index][2], rects[index][3]

    # The {a|b|c} variants: expanded BEFORE the @Name substitution, with the panel's
    # seed (fixed and remembered when needed). So '{@Lea|@Sam}' cites only ONE
    # character: only its refs, LoRAs and negatives go to the engine.
    seed = panel_seed(project, page, panel)
    res = resolve_casting(_expand_text(panel.get("text", ""), seed),
                          _expanded_casting(project.get("casting"), seed))
    style = project.get("style") or {}
    # the order: the panel's text (the cast resolved), the book's style, the mood
    # (chapter > global) - the mood is an atmosphere, never a subject
    parts = [res["prompt"],
             (_expand_text(style.get("prompt_suffix") or "", seed) or "").strip(),
             _expand_text(effective_mood(project, page), seed)]
    prompt = ", ".join(p for p in parts if p)
    negs = [res["negative"], (_expand_text(style.get("negative") or "", seed) or "").strip()]

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
            "rect": rects[index], "seed": seed,
            "unknown": res["unknown"]}


def detail_prompt(project, panel, subject=None):
    """A LOCAL prompt for a detail pass (a face / a hand) on a CROP of the panel.

    It NEVER returns the scene's text: sending the global prompt on a fragment makes the
    crop drift towards the whole scene (a bug fixed four times in this repo -- see the
    detailer's history). We return the description of the character cited first, or
    `subject` when the caller knows better, or an empty string -- an empty prompt is a
    VALID and safe result, not a failure case."""
    if subject:
        return subject.strip()
    # the EXPANDED text (the same seed as the render): in '{@Lea|@Sam}' the subject is
    # the character actually drawn, not the first one cited
    text = _expand_text(panel.get("text") or "", int(panel.get("seed", -1)))
    m = _AT.search(text)
    while m:
        if m.group(0) != "@@":
            char = _casting_get(project.get("casting") or {}, m.group(1))
            # kind 'setting' skipped: 'wide shot of @Castle, @Hero on the ramparts'
            # must detail Hero, not return the castle as the subject of a face.
            if char and char.get("kind", "character") == "character":
                # The same {a|b|c} option as the panel's render (the same seed, the same field).
                return (_expand_text(char.get("desc") or "",
                                     int(panel.get("seed", -1))) or "").strip()
        m = _AT.search(text, m.end())
    return ""


# ----------------------------------------------------------------------------
# Composing the plate
# ----------------------------------------------------------------------------
def _placeholder_font(px):
    """The font of the placeholder label, proportional to the panel. PIL's default
    bitmap is ~10 px: invisible on a plate 2048 px wide."""
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
    """Assembles a PIL plate from its panels' images.

    images : a dict panel_id -> PIL.Image (it has the priority), otherwise we load
             panel['image'].
    fit    : 'cover' = fills the panel and crops at the centre (the default, with no
             empty band); 'contain' = the whole image, the background visible around it.
    placeholders : draws a numbered crossed-out box for the panels not rendered."""
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
    """A multi-page PDF from a list of plate images (the order = the pagination)."""
    imgs = [im.convert("RGB") for im in images if im is not None]
    if not imgs:
        raise ValueError("export_pdf: no page to export")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    imgs[0].save(path, "PDF", resolution=float(dpi), save_all=True,
                 append_images=imgs[1:])
    return path


# ----------------------------------------------------------------------------
# Persistence (project.json)
# ----------------------------------------------------------------------------
def project_json_path(project_dir):
    return os.path.join(project_dir, "project.json")


def save_project(project, project_dir):
    """An ATOMIC write (tmp + os.replace): a crash while writing never leaves a
    truncated project.json -- it is the only place where the script lives."""
    os.makedirs(project_dir, exist_ok=True)
    dst = project_json_path(project_dir)
    tmp = dst + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(project, f, indent=2, ensure_ascii=False)
    # Windows refuses os.replace() on a file that ANOTHER reader holds
    # open (PermissionError WinError 5): a UI poll re-reading
    # project.json while a batch saves it is enough. The window lasts
    # milliseconds -> we retry briefly instead of failing.
    import time
    for attempt in range(40):
        try:
            os.replace(tmp, dst)
            return dst
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)
    return dst


def load_project(project_dir):
    """Re-reads a project and completes the absent keys (tolerant of the files written
    by an earlier version of the schema)."""
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
# Lettering: dialogue, bubbles, captions, onomatopoeia
# ----------------------------------------------------------------------------
# The text is NEVER asked of the model (Z-Image invents letters:
# 'Mendian Station'), it is drawn vectorially AFTER the composition:
# editable without regenerating the image, translatable, sharp in print.
DIALOGUE_KINDS = ("speech", "thought", "caption", "sfx")

# The bubble shapes: round = an ellipse (the classic), rounded = a rectangle with
# rounded corners (compact, for dense reading), angular = a polygon with cut corners
# (a hard, mechanical voice, a shout). The resolution: the line's style > the
# project's style.bubble > 'round'.
BUBBLE_STYLES = ("round", "rounded", "angular")

# The fonts tried in order (Windows then free fallbacks).
_BUBBLE_FONTS = ("comicbd.ttf", "comic.ttf", "segoeui.ttf", "arial.ttf",
                 "DejaVuSans.ttf")
_SFX_FONTS = ("impact.ttf", "arialbd.ttf", "comicbd.ttf", "DejaVuSans-Bold.ttf")


def add_dialogue(panel, text, speaker=None, kind="speech", anchor=None,
                 style=None):
    """Adds a line of dialogue to a panel. `anchor` = (fx, fy) in fractions of the
    panel, what the bubble's tail points at (the default: the speaker's mouth as
    detected, otherwise the bottom of the bubble). `style` = the bubble's shape
    (BUBBLE_STYLES), None = the project's style."""
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
    """The scriptwriter's syntax -> a list of lines, one per line:
         Kira: On y va.                 -> speech (speaker Kira)
         Kira (think): Trop tard.       -> thought
         Rook (angular): The case stays -> speech, a bubble with cut corners
         Kira (think, rounded): ...     -> thought, a rounded rectangle
         CAP: Trois heures plus tot.    -> caption (narration)
         SFX: KRAK                      -> onomatopoeia
       The modifiers in parentheses (cumulative, comma-separated): think/thought =
       a thought; round/rounded/angular = the bubble's shape (BUBBLE_STYLES). An
       unknown parenthesis stays in the speaker's name. A line with no ':' is a
       caption. Empty lines are ignored."""
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
        mh = re.match(r"^(cap|sfx)\s*\(([^)]+)\)$", low)
        if low in ("cap", "sfx") or mh:
            d = {"speaker": "", "text": text,
                 "kind": "caption" if (mh.group(1) if mh else low) == "cap" else "sfx"}
            if mh:
                raw_mods = re.match(r"^(.*?)\s*\(([^)]+)\)$", head).group(2)
                for tok in raw_mods.split(","):
                    tok = tok.strip()
                    if tok.lower() in ("hidden", "off", "muet"):
                        d["hidden"] = True
                    elif tok.lower().startswith("font="):
                        d["font"] = tok[5:].strip()
                    elif tok.lower().startswith("outline="):
                        try:
                            d["outline"] = float(tok[8:])
                        except ValueError:
                            pass
            out.append(d)
        else:
            kind, style, hidden, font, outline = "speech", None, False, None, None
            m = re.match(r"^(.*?)\s*\(([^)]+)\)$", head)
            if m:
                known = True
                k2, s2, h2, f2, o2 = kind, style, hidden, font, None
                for tok in m.group(2).split(","):
                    tok = tok.strip()
                    tl = tok.lower()
                    if tl in ("think", "thought", "pense"):
                        k2 = "thought"
                    elif tl in BUBBLE_STYLES:
                        s2 = tl
                    elif tl in ("hidden", "off", "muet"):
                        h2 = True                 # kept in the script, not lettered
                    elif tl.startswith("font="):
                        f2 = tok[5:].strip() or None
                    elif tl.startswith("outline="):
                        try:
                            o2 = float(tok[8:])
                        except ValueError:
                            known = False
                            break
                    else:
                        known = False
                        break
                if known:
                    head, kind, style, hidden, font = m.group(1).strip(), k2, s2, h2, f2
                    outline = o2
            d = {"speaker": head, "text": text, "kind": kind}
            if style:
                d["style"] = style
            if hidden:
                d["hidden"] = True
            if font:
                d["font"] = font
            if outline is not None:
                d["outline"] = outline
            out.append(d)
    return out


# The fonts offered in the UI (file -> label). Probed on demand: only the ones that
# load are listed. A fonts/ folder next to this module (or next to the project) adds
# its .ttf/.otf - that is where the downloaded comic fonts go (Komika, Anime Ace,
# Blambot...).
FONT_CANDIDATES = (
    ("comicbd.ttf", "Comic Sans Bold"), ("comic.ttf", "Comic Sans"),
    ("segoepr.ttf", "Segoe Print"), ("segoeprb.ttf", "Segoe Print Bold"),
    ("segoesc.ttf", "Segoe Script"), ("segoescb.ttf", "Segoe Script Bold"),
    ("impact.ttf", "Impact"), ("arialbd.ttf", "Arial Bold"), ("arial.ttf", "Arial"),
    ("ariblk.ttf", "Arial Black"), ("georgia.ttf", "Georgia"),
    ("verdana.ttf", "Verdana"), ("trebucbd.ttf", "Trebuchet Bold"),
    ("bahnschrift.ttf", "Bahnschrift"), ("calibri.ttf", "Calibri"),
    ("DejaVuSans.ttf", "DejaVu Sans"), ("DejaVuSans-Bold.ttf", "DejaVu Sans Bold"),
)
_FONT_DIRS = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")]


def _resolve_font(name):
    """A font name -> the path when it lives in a fonts/ folder, otherwise the
    name as it is (PIL looks through the system fonts)."""
    if not name:
        return None
    for d in _FONT_DIRS:
        p = os.path.join(d, os.path.basename(str(name)))
        if os.path.isfile(p):
            return p
    return str(name)


def available_fonts(extra_dirs=()):
    """[{'file', 'label'}] of the fonts usable here: the system candidates that
    load + every .ttf/.otf of the fonts/ folders (the module's, then extra_dirs,
    the book's folder say)."""
    from PIL import ImageFont
    out, seen = [], set()
    dirs = list(_FONT_DIRS) + [d for d in extra_dirs if d]
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".ttf", ".otf")) and f not in seen:
                seen.add(f)
                out.append({"file": f, "label": os.path.splitext(f)[0]})
    for f, label in FONT_CANDIDATES:
        if f in seen:
            continue
        try:
            ImageFont.truetype(f, 20)
        except Exception:
            continue
        seen.add(f)
        out.append({"file": f, "label": label})
    return out


def _font(candidates, px):
    from PIL import ImageFont
    for name in candidates:
        if not name:
            continue
        try:
            return ImageFont.truetype(_resolve_font(name), px)
        except Exception:
            continue
    return ImageFont.load_default()


def manual_breaks(text):
    """A '\\n' typed in a line of dialogue (two characters) = a forced line
    break; the real breaks are kept as they are."""
    return str(text or "").replace("\\n", "\n")


def _wrap(draw, text, font, max_w):
    """Cuts the text into lines fitting in max_w pixels (by words). A forced
    break (\\n in the line of dialogue) always starts a line."""
    lines = []
    for para in manual_breaks(text).split("\n"):
        words, cur = para.split(), ""
        for w in words:
            cand = (cur + " " + w).strip()
            if cur and draw.textlength(cand, font=font) > max_w:
                lines.append(cur)
                cur = w
            else:
                cur = cand
        lines.append(cur)
    return lines or [""]


def _overlap_area(a, b):
    """The intersection area of two rects (x, y, w, h)."""
    ox = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    oy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0, ox) * max(0, oy)


def _face_zone(box, panel_rect, grow=0.15):
    """A face bbox (x1,y1,x2,y2) -> a forbidden rect (x,y,w,h), widened by `grow`
    and bounded by the panel."""
    x1, y1, x2, y2 = box
    gx, gy = (x2 - x1) * grow, (y2 - y1) * grow
    x1, y1, x2, y2 = x1 - gx, y1 - gy, x2 + gx, y2 + gy
    px, py, pw, ph = panel_rect
    x1, y1 = max(px, x1), max(py, y1)
    x2, y2 = min(px + pw, x2), min(py + ph, y2)
    return (int(x1), int(y1), max(0, int(x2 - x1)), max(0, int(y2 - y1)))


def _place_rect(panel_rect, bw, bh, forbidden, taken, prefer):
    """The position of a bw x bh rect in the panel: it sweeps from the top down,
    the columns' order according to `prefer` ('left' / 'right' / 'center').

    The absolute priority: ZERO overlap of the face areas (`forbidden`) and of the
    bubbles already placed (`taken`). Should no clean position exist, it returns the
    one that overlaps the LEAST face (a last resort, never silent: the placement is
    reported as clipped=True in the lettering's return value)."""
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
    """A bubble's MANUAL position: d['pos'] = the top-left corner in FRACTIONS of
    the panel, written by Comic Studio when the user moves the bubble.
    It has the priority over the automatic placement (_place_rect), clamped inside
    the panel. clean=False when it overlaps a face: we do NOT move it again
    (the user put it there on purpose), we only report it.
    Returns ((x, y, w, h), clean) or None when there is no manual position."""
    pos = d.get("pos")
    if not pos:
        return None
    x, y, w, h = panel_rect
    bx = max(x + 2, min(x + int(float(pos[0]) * w), x + w - bw - 2))
    by = max(y + 2, min(y + int(float(pos[1]) * h), y + h - bh - 2))
    r = (bx, by, bw, bh)
    clean = all(_overlap_area(r, f) == 0 for f in forbidden)
    return r, clean


def _thought_steps(dist, fpx):
    """The number of circles of a THOUGHT tail: ~1 circle every 1.6 font bodies
    along the bubble->tip path, bounded to [2, 8]. A distant thought draws a real
    string of them, a thought stuck to the face stays sober."""
    return max(2, min(8, int(round(dist / max(12.0, fpx * 1.6)))))


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
    """The cosine similarity of two vectors (lists of floats)."""
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(x * x for x in b))
    return num / (da * db) if da and db else 0.0


def _match_speakers(speakers, faces, char_embeddings=None, threshold=0.2):
    """Pairs the speakers (lowercase names, in order of citation) with the faces of
    a panel -> {speaker: face}.

    1. RECOGNITION first: when char_embeddings supplies the embedding of a speaker's
       reference portrait and the detected faces carry theirs, a greedy pairing by
       best cosine similarity (>= threshold).
    2. The remaining speakers take the remaining faces in READING ORDER (the 1st
       speaker cited = the leftmost face) - the v1 heuristic, which stays the
       fallback when there are no references."""
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
    """Draws the dialogue on the composed plate (in place) and returns the list of
    placements [{panel, kind, rect, tip, clean}].

    A line of dialogue can carry d['pos'] = [fx, fy] (the top-left corner in fractions
    of the panel, written by Comic Studio on a drag): the bubble is then placed THERE,
    clamped inside the panel, instead of the automatic placement. d['anchor'] (already
    in v1) drives the tail's TIP the same way. d['scale'] (0.4-3.0) is a per-line
    HOMOTHETY: the font, the margins, the tail and the bubble grow together,
    proportions preserved.

    With `face_detector` (a callable image -> [{'box': (x1,y1,x2,y2),
    'mouth': (x,y)|None}], see cz_face.detect_faces_full):
      - the bubbles NEVER overlap a face (the forbidden areas are widened;
        when the panel is too full, the overlap is minimal and clean=False);
      - a bubble's tail aims at its speaker's MOUTH: the speakers are paired
        with the faces in reading order (the 1st speaker = the leftmost face),
        an explicit `anchor` always wins;
      - a speaker with NO face (an off-screen voice, a shout from behind, a narrator)
        gets a generic bubble: the tail towards the nearest panel edge.
    Without a detector: the v1 behaviour (stacked at the top, alternating left/right)."""
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

        # --- the panel's faces (plate coordinates) ---
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
        faces.sort(key=lambda f: f["box"][0])            # reading order
        forbidden = [_face_zone(f["box"], rect) for f in faces]

        # speaker -> face: recognition by embeddings (the cast's reference
        # portraits) then a reading-order fallback
        # d['hidden']: the line stays in the script but is not
        # lettered (the text is already IN the image, or we want the panel mute)
        dialogue = [d for d in dialogue if not d.get("hidden")]
        if not dialogue:
            continue
        speakers = []
        for d in dialogue:
            s = (d.get("speaker") or "").strip().lower()
            if d.get("kind") in ("speech", "thought") and s and s not in speakers:
                speakers.append(s)
        face_of = _match_speakers(speakers, faces, char_embeddings)

        # a font size RELATIVE to the rendered plate (not absolute px):
        # the same bubble keeps the same proportion on the screen (1980 px tall)
        # and in print (3508 px) - before, the 44 px cap made the text
        # twice as small on the print output
        ph = sheet.height
        fpx = max(int(ph * 0.008), min(int(ph * 0.0225), h // 22))
        book_font = (project.get("style") or {}).get("font") or None
        bubble_fonts = ([book_font] if book_font else []) + list(_BUBBLE_FONTS)
        font = _font(bubble_fonts, fpx)
        pad = max(8, fpx // 2)
        taken = []
        side = 0

        for d in dialogue:
            kind = d.get("kind", "speech")
            text = d.get("text") or ""
            # the font: the line's > the book's > the defaults
            d_fonts = ([d["font"]] if d.get("font") else []) + bubble_fonts
            # A per-line homothety (Comic Studio): d['scale'] multiplies the
            # the font -> the bubble AND the text grow together, the same proportions.
            try:
                scale = float(d.get("scale") or 1.0)
            except (TypeError, ValueError):
                scale = 1.0
            scale = max(0.4, min(3.0, scale))
            fpx_d = fpx if scale == 1.0 else max(10, int(round(fpx * scale)))
            font_d = font if (scale == 1.0 and not d.get("font")) \
                else _font(d_fonts, fpx_d)
            pad_d = max(8, fpx_d // 2)
            sfx_fonts = ([d["font"]] if d.get("font") else []) + list(_SFX_FONTS)
            # the outline's thickness (the bubble, the caption, the SFX stroke): x0.3..x3
            try:
                outline_k = float(d.get("outline") or 1.0)
            except (TypeError, ValueError):
                outline_k = 1.0
            outline_k = max(0.3, min(3.0, outline_k))
            bw = max(1, int(round(3 * outline_k)))
            text = manual_breaks(text)
            if kind == "sfx":
                # auto-fit: a long shout or a cover TITLE must not overflow
                # the panel -> the font shrinks until it fits in
                # width (a manual scale loosens/tightens that cap)
                size = int(max(fpx * 2, h // 8) * scale)
                sfx_font = _font(sfx_fonts, size)
                sw = max(1, int(round(max(3, fpx_d // 5) * outline_k)))
                bb = draw.textbbox((0, 0), text, font=sfx_font, stroke_width=sw,
                                   align="center")
                while size > 10 and bb[2] - bb[0] > int(w * 0.92 * scale):
                    size = int(size * 0.85)
                    sfx_font = _font(sfx_fonts, size)
                    bb = draw.textbbox((0, 0), text, font=sfx_font, stroke_width=sw,
                                       align="center")
                bw_, bh_ = bb[2] - bb[0], bb[3] - bb[1]
                (rx, ry, _, _), clean = _pos_rect(d, rect, bw_, bh_, forbidden) \
                    or _place_rect(rect, bw_, bh_, forbidden, taken, "center")
                draw.text((rx, ry), text, font=sfx_font, fill="#ffffff",
                          stroke_width=sw, stroke_fill="#000000", align="center")
                taken.append((rx, ry, bw_, bh_))
                placements.append({"panel": panel["id"], "kind": kind,
                                   "rect": (rx, ry, bw_, bh_), "tip": None,
                                   "clean": clean})
                continue

            # the wrapping width follows the scale (a true homothety: the bubble
            # keeps its proportions), capped by the panel
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
                               fill="#fdf6d8", outline="#000000", width=bw)
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
            if bstyle == "round":     # an ellipse: the text fits in the inscribed rectangle
                bw_ = int((text_w + pad_d * 2) * 1.25)
                bh_ = int((text_h + pad_d * 2) * 1.45)
            else:                     # a rounded rectangle / cut corners: compact
                bw_ = text_w + pad_d * 3
                bh_ = text_h + pad_d * 3
            prefer = "left" if side == 0 else "right"
            spk = (d.get("speaker") or "").strip().lower()
            fc = face_of.get(spk)
            if fc:  # near the speaker: the column on the side of their face
                prefer = "left" if (fc["box"][0] + fc["box"][2]) / 2 < x + w / 2 \
                    else "right"
            (bx0, by0, _, _), clean = _pos_rect(d, rect, bw_, bh_, forbidden) \
                or _place_rect(rect, bw_, bh_, forbidden, taken, prefer)
            cx, cy = bx0 + bw_ // 2, by0 + bh_ // 2

            # the tail's tip: an explicit anchor > the speaker's mouth > an edge
            if d.get("anchor"):
                ax, ay = d["anchor"]
                tip = (max(x + 2, min(x + int(ax * w), x + w - 2)),
                       max(y + 2, min(y + int(ay * h), y + h - 2)))
            elif fc:
                bx1, by1, bx2, by2 = fc["box"]
                # with no keypoints: the mouth is ~at 4/5 of the face's height
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

            # the tail's base: the bubble's edge on the TIP's SIDE (the bottom when
            # the tip is below, the top when it is above the bubble)
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
                                       fill="#ffffff", outline="#000000", width=bw)
            elif bstyle == "angular":
                c = max(6, min(bw_, bh_) // 5)
                pts = [(bx0 + c, by0), (bx0 + bw_ - c, by0),
                       (bx0 + bw_, by0 + c), (bx0 + bw_, by0 + bh_ - c),
                       (bx0 + bw_ - c, by0 + bh_), (bx0 + c, by0 + bh_),
                       (bx0, by0 + bh_ - c), (bx0, by0 + c)]
                draw.polygon(pts, fill="#ffffff")
                draw.line(pts + [pts[0]], fill="#000000", width=bw, joint="curve")
            else:
                draw.ellipse([bx0, by0, bx0 + bw_, by0 + bh_],
                             fill="#ffffff", outline="#000000", width=bw)
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
                # A number of circles PROPORTIONAL to the bubble->tip distance
                # (see _thought_steps), with decreasing radii towards the tip.
                dist = math.hypot(tip[0] - ex, tip[1] - ey)
                n = _thought_steps(dist, fpx_d)
                for i in range(n):
                    k = (i + 1) / (n + 1.0)
                    r = max(2, int(round(fpx_d * (0.34 - 0.20 * k))))
                    px_ = int(ex + (tip[0] - ex) * k)
                    py_ = int(ey + (tip[1] - ey) * k)
                    draw.ellipse([px_ - r, py_ - r, px_ + r, py_ + r],
                                 fill="#ffffff", outline="#000000", width=max(1, int(round(2 * outline_k))))
            ty = by0 + (bh_ - text_h) // 2
            for l in lines:
                lw = draw.textlength(l, font=font_d)
                draw.text((bx0 + (bw_ - lw) // 2, ty), l, font=font_d,
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
def sheet_prompt(char, style=None, seed=-1):
    """(prompt, negative) to generate the reference plate of a cast sheet: a
    canonical portrait (the seed fixed on the caller's side) which then serves as an
    Omni ref. For a setting (kind 'setting'): an empty establishing shot.
    The fields' {a|b|c} groups are expanded with `seed` (-1 = a free draw)."""
    char = _expanded_casting({"_": char}, seed)["_"]
    style = dict(style or {})
    for key in ("prompt_suffix", "negative"):
        if style.get(key):
            style[key] = _expand_text(style[key], seed)
    desc = (char.get("desc") or "").strip()
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
    """A CBZ (the standard format of the comic readers): a zip of numbered images.
    `pages` = PIL images or file paths, in pagination order."""
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
# The render orchestration (the engine is INJECTED: testable without a GPU)
# ----------------------------------------------------------------------------
def render_project(project, project_dir, engine, only=None, force=False,
                   progress=None):
    """Generates the project's panels through `engine(spec) -> PIL.Image`.

    spec = resolve_panel() + {'chapter','page','panel'} (the ids). A panel that
    already has an image is skipped unless `force`. `only` filters by full id
    'ch01.p02.pn3' or by prefix ('ch01', 'ch01.p02'). The image is saved through
    panel_path() and the project.json is updated after EVERY panel (a crash
    halfway loses nothing). Returns the list of the ids rendered."""
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
    """Changes a plate's role in the book (PAGE_ROLES)."""
    if role not in PAGE_ROLES:
        raise ValueError(f"role must be one of {PAGE_ROLES}, got {role!r}")
    page = find_page(project, chapter_id, page_id)
    page["role"] = role
    return page


def book_order(project):
    """(chapter, page) in PUBLICATION ORDER: the 'cover' ones first, then
    title/story chapter by chapter in the document's order, the 'back' ones
    last - even when plates have been added after the back cover. A STABLE sort:
    at an equal rank, the document's order is preserved."""
    flat = [(ch, pg) for ch in project["chapters"] for pg in ch["pages"]]
    return sorted(flat, key=lambda t: _ROLE_RANK.get(t[1].get("role", "story"), 1))


def _draw_page_number(sheet, pg, number):
    """The folio at the bottom centre, in the margin (never over the panels)."""
    draw = ImageDraw.Draw(sheet)
    margin = int(pg.get("margin") or 0)
    fpx = max(14, min(36, margin - 8)) if margin >= 24 else 0
    if not fpx:
        return                                  # no margin = no folio
    font = _font(_BUBBLE_FONTS, fpx)
    text = str(number)
    tw = draw.textlength(text, font=font)
    try:                                        # the ink according to the background's luminance
        from PIL import ImageColor
        r, g, b = ImageColor.getrgb(pg.get("background", "#ffffff"))[:3]
        ink = "#000000" if (0.299 * r + 0.587 * g + 0.114 * b) > 128 else "#e8e8e8"
    except Exception:
        ink = "#000000"
    draw.text(((pg["width"] - tw) // 2, pg["height"] - margin + (margin - fpx) // 2),
              text, font=font, fill=ink)


def compose_book(project, project_dir, letter=True, fit="cover",
                 face_detector=None, char_embeddings=None, numbers=None):
    """Composes and saves the WHOLE book in publication order (book_order):
    covers, chapters, back. Returns the list of paths, ready for
    export_pdf / export_cbz.

    numbers: None = follows project['page']['page_numbers'] (False by default, so as
    not to alter the existing albums); True/False forces it. Only the 'story' plates
    are folioed (1, 2, ...) - covers, flyleaves and back covers never have a number."""
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
    """Composes and saves every plate of a chapter (+ the lettering), returns the
    list of paths in pagination order."""
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
