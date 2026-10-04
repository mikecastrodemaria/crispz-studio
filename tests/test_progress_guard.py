"""Unit tests for the Gradio progress-task guard (no server, no UI).

Regression guard for a long model load: gradio.queueing.Queue.send_message reads
`pending_messages_per_session[event.session_hash]` with no guard, and routes.py deletes
that entry as soon as a session's SSE stream ends. A blocking event outlives the tab that
started it easily enough -- a 13 GB checkpoint converts in ~400s -- and the push then
raises

    KeyError: 'q84y79w8x6q'   (in Queue.start_progress_updates)

That coroutine is a SINGLE long-lived asyncio task, so it dies for good: no progress is
sent again, for any session, until the app is restarted, while the event itself keeps
running to the end. Hence a render that lands in the output folder with a frozen UI.

Run:  .venv/Scripts/python tests/test_progress_guard.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_cli  # noqa: E402
from gradio import queueing  # noqa: E402


class _FakeEvent:
    def __init__(self, session_hash):
        self.session_hash = session_hash
        self.alive = True
        self._id = "evt-1"


class _FakeMsg:
    event_id = None


def _fake_queue(sessions):
    """A stand-in with just what send_message touches."""
    q = types.SimpleNamespace()
    q.pending_messages_per_session = {s: _Recorder() for s in sessions}
    return q


class _Recorder:
    def __init__(self):
        self.items = []

    def put_nowait(self, m):
        self.items.append(m)


def test_unpatched_gradio_still_has_the_hole():
    """If upstream ever guards it, this patch is dead weight -- fail loudly then."""
    src = queueing.Queue.send_message.__doc__ or ""
    fn = getattr(queueing.Queue.send_message, "__wrapped__", queueing.Queue.send_message)
    assert "pending_messages_per_session" in (
        getattr(fn, "__code__", None) and fn.__code__.co_names or ()
    ) or "pending_messages_per_session" in src, "send_message no longer looks the same"


def test_the_guard_drops_an_update_for_a_dead_session():
    cz_cli._keep_progress_task_alive()
    q = _fake_queue(["alive"])
    # the tab that started the event is gone: this must NOT raise
    queueing.Queue.send_message(q, _FakeEvent("gone"), _FakeMsg())
    assert q.pending_messages_per_session["alive"].items == []


def test_a_live_session_still_receives_it():
    cz_cli._keep_progress_task_alive()
    q = _fake_queue(["alive"])
    msg = _FakeMsg()
    queueing.Queue.send_message(q, _FakeEvent("alive"), msg)
    assert q.pending_messages_per_session["alive"].items == [msg]
    assert msg.event_id == "evt-1", "the original behaviour must be preserved"


def test_applying_it_twice_does_not_stack():
    cz_cli._keep_progress_task_alive()
    first = queueing.Queue.send_message
    cz_cli._keep_progress_task_alive()
    assert queueing.Queue.send_message is first, "the patch must be idempotent"


def test_a_dead_event_is_still_skipped():
    """The guard must not resurrect the check Gradio does itself."""
    cz_cli._keep_progress_task_alive()
    q = _fake_queue(["alive"])
    e = _FakeEvent("alive")
    e.alive = False
    queueing.Queue.send_message(q, e, _FakeMsg())
    assert q.pending_messages_per_session["alive"].items == []


if __name__ == "__main__":
    for fn in (test_unpatched_gradio_still_has_the_hole,
               test_the_guard_drops_an_update_for_a_dead_session,
               test_a_live_session_still_receives_it,
               test_applying_it_twice_does_not_stack,
               test_a_dead_event_is_still_skipped):
        fn()
        print(f"OK {fn.__name__}")
    print("All progress-guard tests passed.")
