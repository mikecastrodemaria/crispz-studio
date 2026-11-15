"""Performance presets and config.txt (2026-11-07, ported from crispz-moss).

A config.txt that copied performance_presets from an older sample used to replace the
app's list entirely, hiding every preset added since. Now: the config's entries first
(they win on a name), the app's missing ones appended, said; performance_presets_merge:
false keeps the config's list alone; an unreadable list falls back to the app's, said;
and the app's built-in list IS the sample's.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_ui as U  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIRST = 'Turbo (8 steps)'
SECOND = 'Quality (20 steps)'


def _perf(cfg):
    out = U._performance_from_config(cfg)
    return out[0], out[-1]


def test_the_built_in_list_is_the_samples():
    with open(os.path.join(ROOT, "config-sample.txt"), encoding="utf-8") as f:
        cfg = json.load(f)
    sample = {k: (int(v[0]), float(v[1])) for k, v in cfg["performance_presets"].items()}
    assert U._BUILT_IN_PERFORMANCE == sample, (U._BUILT_IN_PERFORMANCE, sample)
    print("OK test_the_built_in_list_is_the_samples")


def test_a_config_preset_list_keeps_the_apps_newer_presets():
    old = {"performance_presets": {FIRST: list(U._BUILT_IN_PERFORMANCE[FIRST]), "Mine (12 steps)": [12, 1.0]}}
    perf, note = _perf(old)
    assert list(perf)[:2] == [FIRST, "Mine (12 steps)"], list(perf)
    assert SECOND in perf and perf[SECOND] == U._BUILT_IN_PERFORMANCE[SECOND]
    assert f"'{SECOND}'" in note and "performance_presets_merge" in note, note
    perf, note = _perf(dict(old, performance_presets_merge=False))
    assert list(perf) == [FIRST, "Mine (12 steps)"] and note is None
    perf, note = _perf({"performance_presets": {"X": ["a"]}})
    assert perf == U._BUILT_IN_PERFORMANCE and "unreadable" in note
    perf, note = _perf({})
    assert perf == U._BUILT_IN_PERFORMANCE and note is None
    perf, _n = _perf({"performance_presets": {FIRST: [3, 2.0]}})
    assert perf[FIRST] == (3, 2.0)                      # the config wins on a name
    print("OK test_a_config_preset_list_keeps_the_apps_newer_presets")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All performance-preset tests passed.")
