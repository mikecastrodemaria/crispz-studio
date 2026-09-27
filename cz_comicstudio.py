"""crispz-studio - Comic Studio: a comic editing SPA served in the project's folder.

The same mechanics as the Asset Browser (cz_assetbrowser): the HTML page (vanilla JS,
the source in assets/comicstudio/comicstudio.html) is written INTO the project's
folder (studio.html) and served by Gradio through /gradio_api/file=... It talks to
the app through ONE generic endpoint (api_name='comic_studio': op + dir + a JSON
payload -> JSON), stateless like the Gradio accordion: every operation re-reads and
rewrites project.json, so it is compatible with manual edits, the CLI and the
accordion open at the same time.

No torch/GPU import here: the generation engine and the face detector
are INJECTED by cz_ui (studio_api(engine=..., face_detector_factory=...)),
so the module is testable without a GPU, like cz_comic.

The bubble placements (render_lettering's return value) are saved as a sidecar
'<page>.placements.json' next to the composed PNG: the SPA can thus display and
drag the bubbles without recomposing the plate on every opening.

"""

import os
import json
import threading

import cz_comic
from cz_core import HERE, _dbg

STUDIO_FILE = "studio.html"
_HTML_PATH = os.path.join(HERE, "assets", "comicstudio", "comicstudio.html")


def _resolve_dir(d):
    """The same resolution as the Comic accordion: relative = under the app's folder."""
    d = (d or "").strip() or "comics/my-comic"
    return d if os.path.isabs(d) else os.path.join(HERE, d)


def _write_if_changed(path, text):
    """Writes a file served by the SPA, atomically, and only when it changes
    (the same reason as cz_assetbrowser._write_text_if_changed: the page is a
    constant, we only pay the write + antivirus scan after a code update). A
    deliberate local copy: importing cz_assetbrowser would pull the whole Asset
    Browser SPA in for 15 lines."""
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                if f.read() == text:
                    return False
    except Exception:
        pass
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return True


def studio_html(dir_label):
    """The SPA with the project's folder injected (that is the value the page returns
    as it is to the comic_studio endpoint, like the accordion's textbox)."""
    with open(_HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()
    return html.replace("__CZ_DIR__", json.dumps(dir_label or ""))


def open_studio(project_dir, dir_label=""):
    """Writes studio.html into the project's folder and returns its path.
    The project must exist (project.json): we never write an orphan page."""
    d = _resolve_dir(project_dir)
    if not os.path.isfile(cz_comic.project_json_path(d)):
        raise FileNotFoundError(f"no project.json in {d}")
    dst = os.path.join(d, STUDIO_FILE)
    _write_if_changed(dst, studio_html(dir_label))
    return dst


# ----------------------------------------------------------------------------
# The state sent to the SPA
# ----------------------------------------------------------------------------
def _fmt_dialogue(dlg):
    """Lines of dialogue -> the scriptwriter's syntax (the inverse of parse_dialogue),
    keeping the style modifiers ('Rook (angular): ...') so that the editing
    round trip does not lose the bubbles' shape."""
    lines = []
    for x in dlg or []:
        k, t, s = x.get("kind", "speech"), x.get("text", ""), x.get("speaker", "")
        if k == "caption":
            lines.append(f"CAP: {t}")
        elif k == "sfx":
            lines.append(f"SFX: {t}")
        else:
            mods = []
            if k == "thought":
                mods.append("think")
            if x.get("style"):
                mods.append(x["style"])
            lines.append(f"{s} ({', '.join(mods)}): {t}" if mods else f"{s}: {t}")
    return "\n".join(lines)


def _merge_dialogue(old, new):
    """parse_dialogue starts again from the TEXT: the positions placed by dragging
    (anchor/pos) would be lost on every save of the panel. We stick them back onto
    the unchanged lines: the same (kind, speaker, text) first, otherwise the first
    free line of the same (kind, speaker) - a rewritten line thus keeps
    its bubble in place."""
    used = set()
    for nd in new:
        best = None
        for j, od in enumerate(old or []):
            if j in used:
                continue
            if (od.get("kind") == nd.get("kind")
                    and (od.get("speaker") or "") == (nd.get("speaker") or "")):
                if (od.get("text") or "") == (nd.get("text") or ""):
                    best = j
                    break
                if best is None:
                    best = j
        if best is not None:
            used.add(best)
            for k in ("anchor", "pos"):
                if k in old[best] and k not in nd:
                    nd[k] = old[best][k]
    return new


def _placements_path(project_dir, cid, pid):
    return cz_comic.page_path(project_dir, cid, pid, ext="placements.json")


def _rel_url(path, project_dir):
    """A POSIX relative path for the SPA (served from the project's folder),
    or an absolute /gradio_api/file= URL when the file lives elsewhere."""
    try:
        rel = os.path.relpath(path, project_dir)
    except ValueError:                     # another Windows reader
        rel = ".."
    if rel.startswith(".."):
        return "/gradio_api/file=" + os.path.abspath(path).replace("\\", "/")
    return rel.replace("\\", "/")


def _folio_of(project, cid, pid):
    """The page number (folio) of a 'story' plate in publication order,
    None for cover/title/back (never folioed, as in compose_book)."""
    folio = 0
    for ch, pg in cz_comic.book_order(project):
        if pg.get("role", "story") == "story":
            folio += 1
            if ch["id"] == cid and pg["id"] == pid:
                return folio
        elif ch["id"] == cid and pg["id"] == pid:
            return None
    return None


def _page_state(project, project_dir, chapter, page):
    """Everything the SPA has to know about ONE plate: the panels' geometry in
    fractions of the page (for the clickable overlay), the panels + the dialogue, the
    composed image (a relative URL + the mtime for the cache-buster) and the bubble
    placements (the sidecar written at composition time)."""
    pg = cz_comic.page_size(project.get("page"))
    cells = cz_comic.layout_cells(page["layout"])
    rects = cz_comic.panel_rects(cells, pg["width"], pg["height"],
                                 pg["margin"], pg["gutter"])
    W, H = float(pg["width"]), float(pg["height"])
    ppath = cz_comic.page_path(project_dir, chapter["id"], page["id"])
    url, mtime = None, 0
    if os.path.isfile(ppath):
        url = _rel_url(ppath, project_dir)
        mtime = int(os.path.getmtime(ppath))
    placements = None
    sp = _placements_path(project_dir, chapter["id"], page["id"])
    if os.path.isfile(sp):
        try:
            with open(sp, "r", encoding="utf-8") as f:
                placements = json.load(f)
        except Exception as e:
            _dbg(f"comic-studio: placements sidecar unreadable ({sp}): {e}")
    panels = []
    for i, pn in enumerate(page["panels"]):
        img, imt = None, 0
        p = pn.get("image")
        if p and os.path.isfile(p):
            img = _rel_url(p, project_dir)
            imt = int(os.path.getmtime(p))
        panels.append({
            "id": pn["id"], "text": pn.get("text") or "",
            "seed": int(pn.get("seed", -1)), "status": pn.get("status", "draft"),
            "img": img, "img_mtime": imt,
            "dialogue": pn.get("dialogue") or [],
            "dialogue_text": _fmt_dialogue(pn.get("dialogue")),
            "rect": ([rects[i][0] / W, rects[i][1] / H,
                      rects[i][2] / W, rects[i][3] / H]
                     if i < len(rects) else None),
        })
    return {"cid": chapter["id"], "pid": page["id"],
            "chapter": chapter.get("name") or chapter["id"],
            "role": page.get("role", "story"), "layout": page["layout"],
            "folio": _folio_of(project, chapter["id"], page["id"]),
            "url": url, "mtime": mtime,
            "panels": panels, "placements": placements}


def _state(project, project_dir):
    pg = cz_comic.page_size(project.get("page"))
    book = [_page_state(project, project_dir, ch, p)
            for ch, p in cz_comic.book_order(project)]
    return {"ok": True, "name": project.get("name") or "Untitled",
            "page": {"width": pg["width"], "height": pg["height"],
                     "page_numbers": bool(pg.get("page_numbers", False))},
            "casting": sorted(project.get("casting") or {}),
            "layouts": cz_comic.layout_names(),
            "book": book}


# ----------------------------------------------------------------------------
# Composing a plate (+ the placements sidecar)
# ----------------------------------------------------------------------------
def _compose_one(project, project_dir, cid, pid, face_detector=None,
                 char_embeddings=None):
    """Composes ONE plate + the lettering + the folio (the same rules as compose_book:
    only the 'story' plates are folioed, when page_numbers is active),
    saves the PNG and the placements sidecar. Returns the enriched placements
    (fractions of the page + the line index per panel, for the SPA drag)."""
    page = cz_comic.find_page(project, cid, pid)
    pg = cz_comic.page_size(project.get("page"))
    sheet = cz_comic.compose_page(project, page)
    raw = cz_comic.render_lettering(project, page, sheet,
                                    face_detector=face_detector,
                                    char_embeddings=char_embeddings)
    folio = _folio_of(project, cid, pid)
    if folio and bool(pg.get("page_numbers", False)):
        cz_comic._draw_page_number(sheet, pg, folio)
    dst = cz_comic.page_path(project_dir, cid, pid)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    sheet.save(dst)
    W, H = float(pg["width"]), float(pg["height"])
    counters, placements = {}, []
    for pl in raw:                        # one line of dialogue = exactly one placement,
        pnid = pl["panel"]                # in the order of the panel's dialogue ->
        idx = counters.get(pnid, 0)       # the index = the position in panel['dialogue']
        counters[pnid] = idx + 1
        x, y, w, h = pl["rect"]
        placements.append({
            "panel": pnid, "kind": pl["kind"], "index": idx,
            "rect": [x / W, y / H, w / W, h / H],
            "tip": ([pl["tip"][0] / W, pl["tip"][1] / H] if pl.get("tip") else None),
            "clean": bool(pl.get("clean", True))})
    sp = _placements_path(project_dir, cid, pid)
    try:
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(placements, f, ensure_ascii=False)
    except Exception as e:
        _dbg(f"comic-studio: placements sidecar write failed ({sp}): {e}")
    return placements


# ----------------------------------------------------------------------------
# The generic endpoint (api_name='comic_studio')
# ----------------------------------------------------------------------------
def studio_api(op, project_dir, payload="", engine=None,
               face_detector_factory=None, char_embeddings_factory=None):
    """The dispatch of the SPA's operations. It ALWAYS returns a JSON string
    ({ok: true, ...} or {ok: false, error}): the SPA has only one error path.

    op / payload:
      state                                   -> the whole project (the book view)
      save_panel   {cid,pid,pnid,text,dialogue,seed} -> {ok, unknown, page}
      set_bubble   {cid,pid,pnid,index, pos|anchor:[fx,fy] | clear:[...]}
                    -> recomposes the plate, {ok, page}
      compose      {cid,pid}                  -> {ok, page}
      compose_book {}                         -> {ok, state}  (the whole book)
      generate     {cid,pid,pnid}             -> generates the panel (the injected
                                                 engine), recomposes the plate, {ok, page}

    The engine (engine) and the face-aware lettering (factories) are injected by
    cz_ui; absent (the tests, comic disabled), generate fails cleanly and the
    composition letters with no forbidden areas (the v1 behaviour)."""
    try:
        d = _resolve_dir(project_dir)
        data = json.loads(payload) if (payload or "").strip() else {}
        if not isinstance(data, dict):
            raise ValueError("payload must be a JSON object")
        project = cz_comic.load_project(d)

        def _letter_kit():
            fd = face_detector_factory() if face_detector_factory else None
            emb = (char_embeddings_factory(project, d)
                   if (fd and char_embeddings_factory) else None)
            return fd, emb

        def _page_reply(cid, pid, extra=None):
            ch = cz_comic.find_chapter(project, cid)
            page = cz_comic.find_page(project, cid, pid)
            out = {"ok": True, "page": _page_state(project, d, ch, page)}
            out.update(extra or {})
            return json.dumps(out, ensure_ascii=False)

        if op == "state":
            return json.dumps(_state(project, d), ensure_ascii=False)

        if op == "save_panel":
            cid, pid = data["cid"], data["pid"]
            panel = cz_comic.find_panel(project, cid, pid, data["pnid"])
            panel["text"] = (data.get("text") or "").strip()
            old = panel.get("dialogue") or []
            panel["dialogue"] = _merge_dialogue(
                old, cz_comic.parse_dialogue(data.get("dialogue") or ""))
            try:
                panel["seed"] = int(data.get("seed", panel.get("seed", -1)))
            except (TypeError, ValueError):
                pass
            cz_comic.save_project(project, d)
            unknown = cz_comic.resolve_casting(
                panel["text"], project.get("casting"))["unknown"]
            return _page_reply(cid, pid, {"unknown": unknown})

        if op == "set_bubble":
            cid, pid = data["cid"], data["pid"]
            panel = cz_comic.find_panel(project, cid, pid, data["pnid"])
            dlg = panel.get("dialogue") or []
            idx = int(data.get("index", -1))
            if not 0 <= idx < len(dlg):
                raise IndexError(f"dialogue index {idx} out of range "
                                 f"(panel has {len(dlg)} line(s))")
            for key in ("pos", "anchor"):
                if data.get(key) is not None:
                    fx, fy = data[key]
                    dlg[idx][key] = [max(0.0, min(1.0, float(fx))),
                                     max(0.0, min(1.0, float(fy)))]
            for key in data.get("clear") or []:
                if key in ("pos", "anchor"):
                    dlg[idx].pop(key, None)
            cz_comic.save_project(project, d)
            fd, emb = _letter_kit()
            _compose_one(project, d, cid, pid, fd, emb)
            return _page_reply(cid, pid)

        if op == "compose":
            cid, pid = data["cid"], data["pid"]
            fd, emb = _letter_kit()
            _compose_one(project, d, cid, pid, fd, emb)
            return _page_reply(cid, pid)

        if op == "compose_book":
            fd, emb = _letter_kit()
            for ch, pg in cz_comic.book_order(project):
                _compose_one(project, d, ch["id"], pg["id"], fd, emb)
            return json.dumps({"ok": True, "state": _state(project, d)},
                              ensure_ascii=False)

        if op == "generate":
            if engine is None:
                raise RuntimeError("no generation engine (comic disabled?)")
            cid, pid, pnid = data["cid"], data["pid"], data["pnid"]
            done = cz_comic.render_project(project, d, engine,
                                           only=[f"{cid}.{pid}.{pnid}"],
                                           force=True)
            if not done:
                raise RuntimeError(f"panel {cid}.{pid}.{pnid} not rendered")
            fd, emb = _letter_kit()
            _compose_one(project, d, cid, pid, fd, emb)
            return _page_reply(cid, pid, {"rendered": done})

        raise ValueError(f"unknown op '{op}'")
    except Exception as e:
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"},
                          ensure_ascii=False)
