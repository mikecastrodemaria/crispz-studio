"""crispz-studio - Comic Studio: SPA d'edition de BD servie dans le dossier du projet.

Meme mecanique que l'Asset Browser (cz_assetbrowser): la page HTML (vanilla JS,
source dans assets/comicstudio/comicstudio.html) est ecrite DANS le dossier du
projet (studio.html) et servie par Gradio via /gradio_api/file=... Elle parle a
l'app par UN endpoint generique (api_name='comic_studio': op + dir + payload
JSON -> JSON), stateless comme l'accordeon Gradio: chaque operation relit et
reecrit project.json, donc compatible avec les edits manuels, le CLI et
l'accordeon ouverts en meme temps.

Aucun import torch/GPU ici: le moteur de generation et le detecteur de visages
sont INJECTES par cz_ui (studio_api(engine=..., face_detector_factory=...)),
le module se teste donc sans GPU, comme cz_comic.

Les placements de bulles (retour de render_lettering) sont sauves en sidecar
'<page>.placements.json' a cote du PNG compose: la SPA peut ainsi afficher et
faire glisser les bulles sans recomposer la planche a chaque ouverture.
"""

import os
import json
import threading

import cz_comic
from cz_core import HERE, _dbg

STUDIO_FILE = "studio.html"
_HTML_PATH = os.path.join(HERE, "assets", "comicstudio", "comicstudio.html")


def _resolve_dir(d):
    """Meme resolution que l'accordeon Comic: relatif = sous le dossier de l'app."""
    d = (d or "").strip() or "comics/my-comic"
    return d if os.path.isabs(d) else os.path.join(HERE, d)


def _write_if_changed(path, text):
    """Ecrit un fichier servi par la SPA, atomiquement, et seulement s'il change
    (meme raison que cz_assetbrowser._write_text_if_changed: la page est une
    constante, on ne paie le write + scan antivirus qu'apres une mise a jour du
    code). Copie locale volontaire: importer cz_assetbrowser tirerait toute la
    SPA Asset Browser pour 15 lignes."""
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
    """SPA avec le dossier du projet injecte (c'est la valeur que la page renvoie
    telle quelle a l'endpoint comic_studio, comme la textbox de l'accordeon)."""
    with open(_HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()
    return html.replace("__CZ_DIR__", json.dumps(dir_label or ""))


def open_studio(project_dir, dir_label=""):
    """Ecrit studio.html dans le dossier du projet et renvoie son chemin.
    Le projet doit exister (project.json): on n'ecrit jamais de page orpheline."""
    d = _resolve_dir(project_dir)
    if not os.path.isfile(cz_comic.project_json_path(d)):
        raise FileNotFoundError(f"no project.json in {d}")
    dst = os.path.join(d, STUDIO_FILE)
    _write_if_changed(dst, studio_html(dir_label))
    return dst


# ----------------------------------------------------------------------------
# Etat envoye a la SPA
# ----------------------------------------------------------------------------
def _fmt_dialogue(dlg):
    """Repliques -> syntaxe scenariste (l'inverse de parse_dialogue), en
    conservant les modificateurs de style ('Rook (angular): ...') pour que
    l'aller-retour edition ne perde pas la forme des bulles."""
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
    """parse_dialogue repart du TEXTE: les positions posees au drag (anchor/pos)
    seraient perdues a chaque sauvegarde du panneau. On les recolle sur les
    repliques inchangees: meme (kind, speaker, texte) d'abord, sinon premiere
    replique libre du meme (kind, speaker) - une replique reecrite garde ainsi
    sa bulle en place."""
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
    """Chemin relatif POSIX pour la SPA (servie depuis le dossier du projet),
    ou URL absolue /gradio_api/file= si le fichier vit ailleurs."""
    try:
        rel = os.path.relpath(path, project_dir)
    except ValueError:                     # autre lecteur Windows
        rel = ".."
    if rel.startswith(".."):
        return "/gradio_api/file=" + os.path.abspath(path).replace("\\", "/")
    return rel.replace("\\", "/")


def _folio_of(project, cid, pid):
    """Numero de page (folio) d'une planche 'story' dans l'ordre de publication,
    None pour cover/title/back (jamais foliotees, comme compose_book)."""
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
    """Tout ce que la SPA doit savoir sur UNE planche: geometrie des cases en
    fractions de page (pour l'overlay cliquable), panneaux + dialogues, image
    composee (URL relative + mtime pour le cache-buster) et placements de
    bulles (sidecar ecrit a la composition)."""
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
# Composition d'une planche (+ sidecar de placements)
# ----------------------------------------------------------------------------
def _compose_one(project, project_dir, cid, pid, face_detector=None,
                 char_embeddings=None):
    """Compose UNE planche + lettrage + folio (memes regles que compose_book:
    seules les planches 'story' sont foliotees, si page_numbers est actif),
    sauve le PNG et le sidecar de placements. Renvoie les placements enrichis
    (fractions de page + index de replique par panneau, pour le drag SPA)."""
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
    for pl in raw:                        # une replique = exactement un placement,
        pnid = pl["panel"]                # dans l'ordre du dialogue du panneau ->
        idx = counters.get(pnid, 0)       # index = position dans panel['dialogue']
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
# Endpoint generique (api_name='comic_studio')
# ----------------------------------------------------------------------------
def studio_api(op, project_dir, payload="", engine=None,
               face_detector_factory=None, char_embeddings_factory=None):
    """Dispatch des operations de la SPA. Renvoie TOUJOURS une chaine JSON
    ({ok: true, ...} ou {ok: false, error}): la SPA n'a qu'un chemin d'erreur.

    op / payload:
      state                                   -> projet complet (vue livre)
      save_panel   {cid,pid,pnid,text,dialogue,seed} -> {ok, unknown, page}
      set_bubble   {cid,pid,pnid,index, pos|anchor:[fx,fy] | clear:[...]}
                    -> recompose la planche, {ok, page}
      compose      {cid,pid}                  -> {ok, page}
      compose_book {}                         -> {ok, state}  (tout le livre)
      generate     {cid,pid,pnid}             -> genere la case (moteur injecte),
                                                 recompose la planche, {ok, page}

    Le moteur (engine) et le lettrage face-aware (factories) sont injectes par
    cz_ui; absents (tests, comic desactive), generate echoue proprement et la
    composition lettre sans zones interdites (comportement v1)."""
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
