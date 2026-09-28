"""A sampler/schedule change must never swap the scheduler under a running generation.

The scheduler lives on the SHARED pipe, so replacing it is not a local change: a denoise
loop already running keeps its own `timesteps` list but steps whatever `pipe.scheduler`
points at by then. A fresh scheduler knows nothing of those timesteps and has no
begin_index, so diffusers looks the current timestep up, finds nothing, and raises
"IndexError: index 0 is out of bounds for dimension 0 with size 0" in
_init_step_index -> index_for_timestep, halfway through the render.

Met on crispz-krea2 on 2026-09-28: checkpoint switched, then "Apply CivitAI recommended
settings" (which sets the sampler AND the schedule), then Generate. Gradio does not
serialise the events of different listeners, so the two overlap.

What is checked:
  - the lock free -> the swap happens at once (the normal case, unchanged);
  - the lock HELD by another thread -> `pipe.scheduler` is NOT touched, the request is
    recorded, and the status says "on the next run" instead of claiming it is live;
  - the next get_pipe() applies it, which is the path every generation goes through;
  - the same thread holding the lock (the job queue between two jobs) still applies at
    once: an RLock is re-entrant, and that thread is not inside a denoise loop;
  - a schedule the installed diffusers cannot build falls back to Euler rather than
    leaving the pipe without a scheduler.

The schedule used is chosen from what the installed diffusers can really build, not
hard-coded: 'beta' needs scipy, which requirements.txt does not pull, so on a venv without
it (the CI runner, for one) the scheduler refuses to be built and _apply_sampler falls back
to Euler -- correct behaviour, but this test would then be asserting on the fallback.

Neither a GPU nor a model: the pipes are stubs, only the scheduler objects are real.

Run:  .venv/Scripts/python tests/test_sampler_race.py

"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_pipeline as P  # noqa: E402


class _Pipe:
    """Everything _apply_sampler touches on a pipeline: a rebindable `scheduler`."""

    def __init__(self):
        self.scheduler = None


def _buildable_schedule(config):
    """(schedule, its config flag) for a schedule this diffusers build can really make.
    karras and exponential are pure numpy; beta goes through scipy."""
    for name in ("karras", "exponential", "beta"):
        flag = P._SCHEDULE_FLAG[name]
        try:
            sched = P._build_scheduler("euler", name, config)
        except Exception:
            continue
        if getattr(sched.config, flag, False):
            return name, flag
    return None, None


class _State:
    """Saves and restores the module globals this test writes to."""

    NAMES = ("_BASE_PIPE", "_DERIVED", "_BASE_SCHED_CONFIG", "SAMPLER", "SCHEDULE",
             "_SAMPLER_DIRTY", "_ensure_base")

    def __enter__(self):
        self.saved = {n: getattr(P, n) for n in self.NAMES}
        from diffusers import FlowMatchEulerDiscreteScheduler
        self.pipe = _Pipe()
        P._BASE_SCHED_CONFIG = dict(FlowMatchEulerDiscreteScheduler().config)
        P._BASE_PIPE = self.pipe
        P._DERIVED = {"txt2img": self.pipe}
        P.SAMPLER = "euler"
        P.SCHEDULE = "sgm_uniform"
        P._SAMPLER_DIRTY = False
        P._ensure_base = lambda: self.pipe
        P._apply_sampler(self.pipe)          # the starting point: sgm_uniform applied
        assert self.pipe.scheduler is not None, "the stub config builds no scheduler"
        self.schedule, self.flag = _buildable_schedule(P._BASE_SCHED_CONFIG)
        assert self.schedule, "this diffusers build makes none of karras/exponential/beta"
        return self

    def __exit__(self, *exc):
        for n, v in self.saved.items():
            setattr(P, n, v)
        return False


class _Busy:
    """Another thread holding _GPU_LOCK, exactly like a render in progress."""

    def __enter__(self):
        self.held = threading.Event()
        self.release = threading.Event()

        def hold():
            with P._GPU_LOCK:
                self.held.set()
                self.release.wait(10)

        self.t = threading.Thread(target=hold, daemon=True)
        self.t.start()
        assert self.held.wait(10), "the holding thread never took the lock"
        return self

    def __exit__(self, *exc):
        self.release.set()
        self.t.join(10)
        return False


def test_schedule_applies_at_once_when_no_generation_runs():
    with _State() as st:
        before = st.pipe.scheduler
        out = P.set_schedule(st.schedule)
        assert st.pipe.scheduler is not before, "the scheduler was not rebuilt"
        assert getattr(st.pipe.scheduler.config, st.flag) is True, st.pipe.scheduler.config
        assert P._SAMPLER_DIRTY is False, "nothing should be pending"
        assert out == f"Sampler: euler / {st.schedule}", out


def test_schedule_does_not_touch_the_pipe_while_a_generation_holds_the_gpu():
    with _State() as st, _Busy():
        before = st.pipe.scheduler
        out = P.set_schedule(st.schedule)
        # THE regression: rebinding pipe.scheduler here is what crashed the render.
        assert st.pipe.scheduler is before, "the scheduler was swapped under a render"
        assert P._SAMPLER_DIRTY is True, "the change was neither applied nor recorded"
        assert P.SCHEDULE == st.schedule, P.SCHEDULE
        assert out == f"Sampler: euler / {st.schedule} — on the next run", out


def test_the_deferred_change_lands_on_the_next_get_pipe():
    with _State() as st:
        with _Busy():
            P.set_schedule(st.schedule)
            before = st.pipe.scheduler
            assert st.pipe.scheduler is before
        # The render is over: the next generation goes through get_pipe() and picks it up.
        got = P.get_pipe("txt2img")
        assert got is st.pipe, got
        assert st.pipe.scheduler is not before, "the deferred change never landed"
        assert getattr(st.pipe.scheduler.config, st.flag) is True, st.pipe.scheduler.config
        assert P._SAMPLER_DIRTY is False, "the pending flag was not cleared"
        assert P._sampler_status() == f"Sampler: euler / {st.schedule}", P._sampler_status()


def test_the_thread_that_holds_the_lock_still_applies_at_once():
    """The job queue calls the setters between two jobs from the thread that holds the
    lock. An RLock lets it through, and it must: that thread is between two renders."""
    with _State() as st:
        with P._GPU_LOCK:
            before = st.pipe.scheduler
            P.set_schedule(st.schedule)
            assert st.pipe.scheduler is not before, "a re-entrant call was deferred"
            assert P._SAMPLER_DIRTY is False
            assert getattr(st.pipe.scheduler.config, st.flag) is True


def test_sampler_change_follows_the_same_rule():
    with _State() as st, _Busy():
        before = st.pipe.scheduler
        out = P.set_sampler("unipc")
        assert st.pipe.scheduler is before, "the sampler swap hit a running render"
        assert P._SAMPLER_DIRTY is True
        assert out == "Sampler: unipc / sgm_uniform — on the next run", out


def test_an_unchanged_value_is_a_no_op():
    """Re-firing the same value (the UI chains do it on every preset load) must not
    rebuild anything, nor flag a pending change during a render."""
    with _State() as st, _Busy():
        before = st.pipe.scheduler
        out = P.set_schedule("sgm_uniform")
        assert st.pipe.scheduler is before
        assert P._SAMPLER_DIRTY is False, "an unchanged value should pend nothing"
        assert out == "Sampler: euler / sgm_uniform", out


def test_a_schedule_this_build_cannot_make_falls_back_instead_of_breaking():
    """'beta' needs scipy, absent from requirements.txt. Whether the build can make it or
    not, the pipe must come out with a WORKING scheduler -- never None, never half-set.
    That is what _apply_sampler's fallback is for, and it is the difference between a
    schedule quietly reverting to Euler and a render dying at step one."""
    with _State() as st:
        P.set_schedule("beta")
        sched = st.pipe.scheduler
        assert sched is not None, "the pipe was left without a scheduler"
        assert hasattr(sched, "set_timesteps"), sched
        assert P._scheduler_accepts_sigmas(sched), "the fallback cannot take custom sigmas"
        # Applied when the build allows it, silently Euler when it does not.
        assert sched.config.use_beta_sigmas in (True, False), sched.config


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} sampler race tests passed.")
