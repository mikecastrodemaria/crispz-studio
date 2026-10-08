"""Unit tests for the job queue pure helpers (no Gradio event needed).

Run:  .venv/Scripts/python tests/test_queue.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_ui  # noqa: E402


def _stub_vals(prompt="a cat", use_input=False, w=1024, h=768, steps=8, n=2, seed=42):
    """36-slot stand-in for _gen_inputs values, with the indexed slots filled."""
    vals = [None] * 36
    vals[cz_ui._Q_IDX["prompt"]] = prompt
    vals[cz_ui._Q_IDX["use_input"]] = use_input
    vals[cz_ui._Q_IDX["width"]] = w
    vals[cz_ui._Q_IDX["height"]] = h
    vals[cz_ui._Q_IDX["gen_steps"]] = steps
    vals[cz_ui._Q_IDX["image_number"]] = n
    vals[cz_ui._Q_IDX["seed"]] = seed
    return vals


def test_label():
    ms = {"base_repo": "Tongyi-MAI/Z-Image-Turbo", "transformer": None}
    lbl = cz_ui._q_label(_stub_vals(), ms)
    assert "txt2img" in lbl and "Z-Image-Turbo" in lbl and "1024x768" in lbl
    assert "8 steps" in lbl and "seed 42" in lbl and "x2" in lbl and "a cat" in lbl
    # transformer wins over base repo; img2img mode; long prompt truncated
    ms2 = {"base_repo": "x", "transformer": "D:/models/juggernaut_z.safetensors"}
    lbl2 = cz_ui._q_label(_stub_vals(prompt="p" * 80, use_input=True), ms2)
    assert "img2img" in lbl2 and "juggernaut_z.safetensors" in lbl2 and "…" in lbl2


def _labels(items):
    return [i["label"] for i in items]


def test_move():
    # _q_move_many mutates the list IN PLACE (on purpose): _ui_queue_run holds a reference
    # to that state object, so a reordering has to be visible to it. A fresh list per
    # case rather than expecting a pure function.
    def fresh():
        return [{"label": "a"}, {"label": "b"}, {"label": "c"}]

    items = fresh()
    out, sel = cz_ui._q_move_many(items, [2], -1)
    assert _labels(out) == ["a", "c", "b"] and sel == [1]
    assert out is items, "it must mutate the shared object, not hand back a copy"

    items = fresh()
    out, sel = cz_ui._q_move_many(items, [0], -1)    # top edge: unchanged
    assert _labels(out) == ["a", "b", "c"] and sel == [0]

    items = fresh()
    out, sel = cz_ui._q_move_many(items, None, 1)    # nothing ticked
    assert sel == [] and _labels(out) == ["a", "b", "c"]

    items = fresh()
    out, sel = cz_ui._q_move_many(items, [9], -1)    # out of bounds: ignored
    assert sel == [] and _labels(out) == ["a", "b", "c"]


def test_moving_several_ticked_jobs_keeps_their_order_and_stops_at_the_edge():
    """Mike's rule: everything ticked moves one step. The selection follows the jobs --
    pressing Up twice has to move the same block twice -- and a block already against the
    top simply stops instead of scrambling against itself."""
    items = [{"label": c} for c in "abcd"]
    out, sel = cz_ui._q_move_many(items, [1, 2], -1)          # a contiguous block up
    assert _labels(out) == ["b", "c", "a", "d"] and sel == [0, 1]
    out, sel = cz_ui._q_move_many(out, sel, -1)               # already at the top
    assert _labels(out) == ["b", "c", "a", "d"] and sel == [0, 1]

    items = [{"label": c} for c in "abcde"]
    out, sel = cz_ui._q_move_many(items, [0, 3], +1)          # non-contiguous, down
    assert _labels(out) == ["b", "a", "c", "e", "d"] and sel == [1, 4]

    items = [{"label": c} for c in "abc"]
    out, sel = cz_ui._q_move_many(items, [0, 1, 2], +1)       # all of them: nothing moves
    assert _labels(out) == ["a", "b", "c"] and sel == [0, 1, 2]


def test_remove():
    items = [{"label": "a"}, {"label": "b"}, {"label": "c"}]
    out, sel = cz_ui._q_remove_many(items, [1])
    assert _labels(out) == ["a", "c"] and sel == []
    out, sel = cz_ui._q_remove_many(out, [1])
    assert _labels(out) == ["a"] and sel == []
    out, sel = cz_ui._q_remove_many(out, [0])
    assert out == [] and sel == []
    out, sel = cz_ui._q_remove_many([], None)
    assert out == [] and sel == []


def test_removing_several_ticked_jobs_takes_one_click():
    """Five of eight used to be ten clicks. The selection comes back EMPTY on purpose: a
    second Remove must not delete a job nobody ticked."""
    items = [{"label": c} for c in "abcde"]
    out, sel = cz_ui._q_remove_many(items, [0, 2, 4])
    assert _labels(out) == ["b", "d"], _labels(out)
    assert sel == [], sel
    assert out is items, "it must mutate the shared object, not hand back a copy"


def test_render():
    upd, md, btn = cz_ui._q_render([])
    assert "empty" in md and btn["value"] == "+ Queue (0)"
    items = [{"label": "j1"}, {"label": "j2"}]
    upd, md, btn = cz_ui._q_render(items, 1)
    # The jobs live in the CHOICES now, not in the Markdown: the list you read and the
    # thing you tick are one widget. The Markdown is a one-line summary.
    labels = [c[0] for c in upd["choices"]]
    assert labels == ["#1 ▶ j1", "#2 j2"], labels
    assert "2 job(s)" in md and "1. j1" not in md, md
    # A bare index still works: the callers that touch a single job were not changed.
    assert btn["value"] == "+ Queue (2)" and upd["value"] == [1]
    assert cz_ui._q_render(items, [0, 1])[0]["value"] == [0, 1]
    upd, _, _ = cz_ui._q_render(items, 99)           # out of bounds: dropped
    assert upd["value"] == []
    assert cz_ui._q_render(items, [1, 99])[0]["value"] == [1], "a bad index must not poison"
    assert cz_ui._q_render(items)[0]["value"] == []


def test_a_restored_queue_is_rendered_at_build_time():
    """A restart used to show "Job queue (2 restored)" and "+ Queue (2)" above an EMPTY
    list: the components were built empty and only an interaction ever filled them. The
    panel and _q_render now go through the same two helpers, so they cannot drift."""
    items = [{"label": "a"}, {"label": "b"}]
    assert cz_ui._q_choices([]) == []
    assert "empty" in cz_ui._q_summary([]).lower()
    assert cz_ui._q_choices(items) == [("#1 ▶ a", 0), ("#2 b", 1)]
    assert "2 job(s)" in cz_ui._q_summary(items)
    # what the panel is built with == what an update sends
    upd, md, _btn = cz_ui._q_render(items)
    assert upd["choices"] == cz_ui._q_choices(items), upd["choices"]
    assert md == cz_ui._q_summary(items), md


def test_a_page_load_re_seeds_the_queue_from_disk():
    """The module-level snapshot is read once, when build_ui runs, and gr.State hands each
    session a COPY of it -- so clearing the queue and reloading the page brought the
    cleared jobs back while queue.json said 0. A page load reads the file, which
    _q_persist rewrites on every mutation."""
    real = cz_ui._q_load
    try:
        cz_ui._q_load = lambda: [{"label": "from disk"}]
        items, upd, md, btn = cz_ui._ui_queue_reload()
        assert [it["label"] for it in items] == ["from disk"], items
        assert upd["choices"] == [("#1 ▶ from disk", 0)], upd["choices"]
        assert "1 job(s)" in md and btn["value"] == "+ Queue (1)"
        cz_ui._q_load = lambda: []
        items, upd, md, btn = cz_ui._ui_queue_reload()
        assert items == [] and upd["choices"] == [] and "empty" in md.lower()
    finally:
        cz_ui._q_load = real


def test_the_run_next_marker_follows_the_queue_not_the_selection():
    """'▶' marks the head of the queue. Selecting job 2 to move it must not move the
    marker: what runs next and what you are editing are different things."""
    items = [{"label": "a"}, {"label": "b"}, {"label": "c"}]
    for sel in (None, [0], [2], [0, 1, 2]):
        labels = [c[0] for c in cz_ui._q_render(items, sel)[0]["choices"]]
        assert labels[0].startswith("#1 ▶ "), labels
        assert all("▶" not in l for l in labels[1:]), labels
    # and it follows a reorder: the job moved to the head becomes the one marked
    cz_ui._q_move_many(items, [2], -1)
    cz_ui._q_move_many(items, [1], -1)
    labels = [c[0] for c in cz_ui._q_render(items)[0]["choices"]]
    assert labels[0] == "#1 ▶ c", labels


def test_model_state_roundtrip_keys():
    ms = cz_ui._q_model_state()
    assert set(ms) == {"base_repo", "transformer", "loras", "sampler", "schedule",
                       "text_encoder"}



# ---------------------------------------------------- pause / stop semantics ---

def _fake_jobs(n):
    return [{"label": f"j{i + 1}", "ms": {}, "vals": _stub_vals()} for i in range(n)]


def _run_with(stub_generate):
    """Runs _ui_queue_run with a stubbed _ui_generate, touching neither the model
    NOR the queue.json on disk (the user's own instance uses it)."""
    import cz_pipeline
    saved = (cz_ui._ui_generate, cz_ui._q_restore_model_state, cz_ui._q_persist,
             cz_pipeline._STOP, cz_ui._QUEUE_PAUSE)
    ran = []
    try:
        cz_ui._ui_generate = stub_generate
        cz_ui._q_restore_model_state = lambda ms: None
        cz_ui._q_persist = lambda items: None
        cz_pipeline._STOP = False
        items = _fake_jobs(3)
        out = cz_ui._ui_queue_run(items, [])
        return items, out
    finally:
        (cz_ui._ui_generate, cz_ui._q_restore_model_state, cz_ui._q_persist,
         cz_pipeline._STOP, cz_ui._QUEUE_PAUSE) = saved


def test_pause_finishes_current_job_then_halts():
    calls = []

    def gen(*vals, progress=None):
        calls.append(1)
        if len(calls) == 1:                       # pause asked for DURING job 1
            cz_ui._QUEUE_PAUSE = True
        return [], "ok", [], []

    items, out = _run_with(gen)
    assert len(calls) == 1, "pause must let the current job FINISH, then halt"
    assert [j["label"] for j in items] == ["j2", "j3"], \
        "the finished job leaves the queue; the rest stays"
    assert "paused" in out[-3].lower()


def test_stop_keeps_the_interrupted_job_queued():
    import cz_pipeline
    calls = []

    def gen(*vals, progress=None):
        calls.append(1)
        if len(calls) == 1:                       # Stop in the middle of job 1
            cz_pipeline._STOP = True
        return [], "interrupted", [], []

    items, out = _run_with(gen)
    assert len(calls) == 1
    assert [j["label"] for j in items] == ["j1", "j2", "j3"], \
        "an interrupted job must STAY at the head of the queue (it did not finish)"
    assert "interrupted job stays queued" in out[-3]


def test_without_pause_or_stop_the_queue_drains():
    def gen(*vals, progress=None):
        return [], "ok", [], []

    items, out = _run_with(gen)
    assert items == [] and "done: 3 job(s)" in out[-3]


def test_request_pause_sets_the_flag_and_reports():
    saved = cz_ui._QUEUE_PAUSE
    try:
        cz_ui._QUEUE_PAUSE = False
        msg = cz_ui._q_request_pause()
        assert cz_ui._QUEUE_PAUSE is True
        assert "Pause requested" in msg
    finally:
        cz_ui._QUEUE_PAUSE = saved


if __name__ == "__main__":
    # Discovered, like the rest of the suite. This file used to list its tests by hand and
    # a test added above and forgotten in that tuple was defined and never run -- which
    # happened twice while this file was being worked on, each time silently.
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} queue tests passed.")
