"""Fleet-update quiet gate (gateway/fleet_quiet.py) — predicate, durable FIFO, dedup, bounds,
one notice per channel, crash-safe ordered replay, admission placement, API-server 503."""
import asyncio
import json
import os
import stat
import time
from types import SimpleNamespace

import pytest

import gateway.run as run_mod  # import at collection: bootstrap probes the checkout's git dir
from gateway import fleet_quiet
from gateway.run import GatewayRunner
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_FLEET_QUIET_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "iris"))
    return tmp_path


def _marker(root, **kw):
    (root / fleet_quiet.MARKER_NAME).write_text(json.dumps(kw))


def _quiet(root, phase="update", owner=0, ttl=60):
    _marker(root, owner_pid=owner, expires=time.time() + ttl, since=time.time(), phase=phase)


def _ev(text, chat="c1", plat=Platform.TELEGRAM, mid=None):
    return MessageEvent(text=text, source=SessionSource(platform=plat, chat_id=chat, chat_type="dm"),
                        message_id=text if mid is None else mid)


def _dead_pid() -> int:
    pid = 4_000_000
    while True:
        try:
            os.kill(pid, 0)
            pid += 1
        except ProcessLookupError:
            return pid
        except PermissionError:
            pid += 1


# ── marker predicate ─────────────────────────────────────────────────────────────────────────────

class TestPredicate:
    def test_absent_is_release(self, root):
        assert not fleet_quiet.is_active()

    def test_ownerless_passed_unexpired_is_quiet(self, root):
        _quiet(root, phase="passed", owner=0)
        assert fleet_quiet.is_active()

    def test_dead_owner_passed_unexpired_is_still_quiet(self, root):
        _quiet(root, phase="passed", owner=_dead_pid())
        assert fleet_quiet.is_active()

    def test_live_owner_update_is_quiet(self, root):
        _quiet(root, phase="update", owner=os.getpid())
        assert fleet_quiet.is_active()

    @pytest.mark.parametrize("phase", ["preflight", "update"])
    def test_dead_owner_preflight_or_update_is_stale_release(self, root, phase):
        _quiet(root, phase=phase, owner=_dead_pid())
        assert fleet_quiet.evaluate(fleet_quiet.read_marker()) == (False, "owner-dead")

    def test_expiry_is_release_even_with_live_owner(self, root):
        _marker(root, owner_pid=os.getpid(), expires=time.time() - 1, phase="update")
        assert fleet_quiet.evaluate(fleet_quiet.read_marker()) == (False, "expired")

    def test_corrupt_fails_open(self, root):
        (root / fleet_quiet.MARKER_NAME).write_text("{nope")
        assert not fleet_quiet.is_active()

    def test_non_object_fails_open(self, root):
        (root / fleet_quiet.MARKER_NAME).write_text("[1,2]")
        assert not fleet_quiet.is_active()

    def test_undated_clamped(self, root):
        _marker(root, owner_pid=0, since=time.time() - 3 * 3600)
        assert not fleet_quiet.is_active()
        _marker(root, owner_pid=0, since=time.time() - 60)
        assert fleet_quiet.is_active()

    def test_extra_parent_fields_ignored(self, root):
        _marker(root, owner_pid=0, expires=time.time() + 60, paused=["iris"], count=3, pending=[1])
        assert fleet_quiet.is_active()

    def test_profile_home_resolves_shared_root(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_FLEET_QUIET_ROOT", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "athena"))
        assert fleet_quiet.marker_path() == tmp_path / fleet_quiet.MARKER_NAME
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert fleet_quiet.marker_path() == tmp_path / fleet_quiet.MARKER_NAME

    def test_retry_after_clamped(self, root):
        assert fleet_quiet.retry_after_s({"expires": time.time() + 3600}) == 60
        assert fleet_quiet.retry_after_s({"expires": time.time() - 5}) == 1


# ── durable queue ────────────────────────────────────────────────────────────────────────────────

class TestQueue:
    def test_fifo_and_first_flag_per_channel(self, root):
        assert fleet_quiet.hold(_ev("a"))["first"] is True
        assert fleet_quiet.hold(_ev("b"))["first"] is False
        assert fleet_quiet.hold(_ev("x", chat="c2", plat=Platform.SLACK))["first"] is True
        assert [r["text"] for r in fleet_quiet.load_held()] == ["a", "b", "x"]

    def test_duplicate_redelivery_not_appended(self, root):
        fleet_quiet.hold(_ev("a", mid="m1"))
        assert fleet_quiet.hold(_ev("a again", mid="m1")) == {"first": False, "duplicate": True}
        # same id on ANOTHER channel is a different message
        assert fleet_quiet.hold(_ev("a", chat="c9", mid="m1"))["duplicate"] is False
        assert len(fleet_quiet.load_held()) == 2

    def test_bound_refuses(self, root, monkeypatch):
        monkeypatch.setattr(fleet_quiet, "MAX_HELD", 2)
        fleet_quiet.hold(_ev("1"))
        fleet_quiet.hold(_ev("2"))
        with pytest.raises(fleet_quiet.QueueFull):
            fleet_quiet.hold(_ev("3"))

    def test_byte_bound_refuses(self, root, monkeypatch):
        monkeypatch.setattr(fleet_quiet, "MAX_QUEUE_BYTES", 2000)
        fleet_quiet.hold(_ev("small"))
        with pytest.raises(fleet_quiet.QueueFull):
            fleet_quiet.hold(_ev("x" * 2000))

    def test_permissions_private(self, root):
        fleet_quiet.hold(_ev("secret"))
        qd = root / fleet_quiet.QUEUE_DIR_NAME
        assert stat.S_IMODE(qd.stat().st_mode) == 0o700
        assert stat.S_IMODE((qd / "iris.jsonl").stat().st_mode) == 0o600

    def test_round_trip_event_fields(self, root):
        ev = _ev("hi", plat=Platform.SLACK)
        ev.source.thread_id = "t1"
        ev.media_urls, ev.media_types = ["/tmp/x.png"], ["image/png"]
        ev.metadata = {"k": "v", "bad": object()}
        fleet_quiet.hold(ev)
        back = fleet_quiet.deserialize_event(fleet_quiet.load_held()[0])
        assert back.source.platform == Platform.SLACK and back.source.thread_id == "t1"
        assert back.media_urls == ["/tmp/x.png"] and back.metadata["k"] == "v" and "bad" not in back.metadata
        assert back.metadata["fleet_quiet_replay"] is True

    def test_corrupt_line_skipped(self, root):
        fleet_quiet.hold(_ev("a"))
        with open(fleet_quiet.queue_path(), "a") as fh:
            fh.write("{torn\n")
        fleet_quiet.hold(_ev("b"))
        assert [r["text"] for r in fleet_quiet.load_held()] == ["a", "b"]

    def test_claim_and_cursor(self, root):
        for t in "abc":
            fleet_quiet.hold(_ev(t))
        claims = fleet_quiet.claim_held()
        assert len(claims) == 1 and not fleet_quiet.queue_path().exists()
        fleet_quiet.write_cursor(claims[0], 1)
        assert [r["text"] for r in fleet_quiet.load_held()] == ["b", "c"]
        fleet_quiet.finish_claim(claims[0])
        assert fleet_quiet.load_held() == []

    def test_status_file(self, root):
        _quiet(root)
        fleet_quiet.hold(_ev("a"))
        snap = fleet_quiet.write_status()
        assert snap["active"] and snap["held"] == 1 and snap["profile"] == "iris"
        assert json.loads(fleet_quiet.status_path().read_text())["held"] == 1


# ── runner: hold / notice / replay ──────────────────────────────────────────────────────────────

class _Adapter:
    def __init__(self, sink):
        self.sink = sink
        self._active_sessions = {}

    def _event_session_key(self, ev):
        return f"{ev.source.platform.value}:{ev.source.chat_id}"

    async def handle_message(self, ev):
        self.sink.append(ev.text)


def _runner():
    from gateway.run_shutdown import GatewayShutdownMixin
    r = object.__new__(type("R", (GatewayShutdownMixin,), {}))
    r.sent, r.handled = [], []
    r.adapter = _Adapter(r.handled)
    r.online = True
    r._intake_adapter_for = lambda src: r.adapter if r.online else None
    r._delivery_adapter_for = lambda src: r.adapter if r.online else None

    async def _send(event, adapter, content, **kw):
        r.sent.append((event.source.chat_id, content))
    r._send_busy_reply = _send
    return r


def test_hold_one_notice_per_channel_then_ordered_replay(root):
    r = _runner()
    _quiet(root, phase="preflight")

    async def go():
        await r._fleet_quiet_tick()
        assert r._fleet_quiet_should_hold()
        for t, c in (("1", "c1"), ("2", "c1"), ("3", "c2"), ("4", "c1")):
            await r._fleet_quiet_hold(_ev(t, chat=c))
        assert r.sent == [("c1", fleet_quiet.NOTICE_TEXT), ("c2", fleet_quiet.NOTICE_TEXT)]
        assert r.handled == []
        (root / fleet_quiet.MARKER_NAME).unlink()
        await r._fleet_quiet_tick()
        assert r.handled == ["1", "2", "3", "4"]
        assert not r._fleet_quiet_active and fleet_quiet.load_held() == []

    asyncio.run(go())


def test_duplicate_redelivery_gets_no_second_notice_or_replay(root):
    r = _runner()
    _quiet(root)

    async def go():
        await r._fleet_quiet_hold(_ev("a", mid="m1"))
        await r._fleet_quiet_hold(_ev("a", mid="m1"))
        (root / fleet_quiet.MARKER_NAME).unlink()
        await r._fleet_quiet_tick()

    asyncio.run(go())
    assert len(r.sent) == 1 and r.handled == ["a"]


def test_replay_after_restart_with_marker_gone(root):
    fleet_quiet.hold(_ev("left-over"))
    r = _runner()  # fresh process: _fleet_quiet_active never set
    asyncio.run(r._fleet_quiet_tick())
    assert r.handled == ["left-over"]


def test_boot_with_live_update_owner_stays_quiet(root):
    fleet_quiet.hold(_ev("held"))
    _quiet(root, phase="update", owner=os.getpid())
    r = _runner()
    asyncio.run(r._fleet_quiet_tick())
    assert r._fleet_quiet_active and r.handled == []


def test_boot_with_stale_dead_owner_releases(root):
    fleet_quiet.hold(_ev("held"))
    _quiet(root, phase="update", owner=_dead_pid())
    r = _runner()
    asyncio.run(r._fleet_quiet_tick())
    assert not r._fleet_quiet_active and r.handled == ["held"]


def test_replay_waits_for_offline_adapter(root):
    fleet_quiet.hold(_ev("a"))
    fleet_quiet.hold(_ev("b"))
    r = _runner()
    r.online = False
    asyncio.run(r._fleet_quiet_tick())
    assert r.handled == [] and [x["text"] for x in fleet_quiet.load_held()] == ["a", "b"]
    r.online = True
    asyncio.run(r._fleet_quiet_tick())
    assert r.handled == ["a", "b"]


def test_replay_does_not_interrupt_running_session(root):
    """The second held message of a chat waits until the first replayed turn finishes."""
    fleet_quiet.hold(_ev("a"))
    fleet_quiet.hold(_ev("b"))
    r = _runner()

    async def start_turn(ev):
        r.handled.append(ev.text)
        r.adapter._active_sessions[r.adapter._event_session_key(ev)] = True
    r.adapter.handle_message = start_turn
    asyncio.run(r._fleet_quiet_tick())
    assert r.handled == ["a"]
    r.adapter._active_sessions.clear()
    asyncio.run(r._fleet_quiet_tick())
    assert r.handled == ["a", "b"]


def test_stuck_channel_does_not_block_other_channels(root):
    """Photon offline must not silence Telegram: per-channel FIFO, stuck channel carried."""
    fleet_quiet.hold(_ev("p1", chat="p", plat=Platform("photon")))
    fleet_quiet.hold(_ev("t1", chat="t"))
    fleet_quiet.hold(_ev("p2", chat="p", plat=Platform("photon")))
    fleet_quiet.hold(_ev("t2", chat="t"))
    r = _runner()
    photon_up = {"v": False}
    r._intake_adapter_for = lambda src: r.adapter if (src.platform.value != "photon" or photon_up["v"]) else None
    r._delivery_adapter_for = r._intake_adapter_for
    asyncio.run(r._fleet_quiet_tick())
    assert r.handled == ["t1", "t2"]
    assert [x["text"] for x in fleet_quiet.load_held()] == ["p1", "p2"]
    assert r._fleet_quiet_backlog is True
    photon_up["v"] = True
    asyncio.run(r._fleet_quiet_tick())
    assert r.handled == ["t1", "t2", "p1", "p2"]
    assert r._fleet_quiet_backlog is False
    assert list((root / fleet_quiet.QUEUE_DIR_NAME).glob("*.replay-*")) == []


def test_inbound_during_backlog_queues_behind_without_notice(root):
    fleet_quiet.hold(_ev("old"))
    r = _runner()
    r.online = False
    asyncio.run(r._fleet_quiet_tick())  # released, but adapter offline: backlog remains
    assert r._fleet_quiet_should_hold()

    async def go():
        await r._fleet_quiet_hold(_ev("new"))
    asyncio.run(go())
    assert r.sent == []  # not quiet any more: no "update in progress" notice
    r.online = True
    asyncio.run(r._fleet_quiet_tick())
    assert r.handled == ["old", "new"]


def test_crash_mid_replay_resumes_without_duplicates(root):
    for t in "abc":
        fleet_quiet.hold(_ev(t))
    claims = fleet_quiet.claim_held()
    fleet_quiet.write_cursor(claims[0], 2)  # crashed after dispatching a, b
    r = _runner()
    asyncio.run(r._fleet_quiet_tick())
    assert r.handled == ["c"]
    assert list((root / fleet_quiet.QUEUE_DIR_NAME).glob("*.replay-*")) == []


def test_reengage_mid_replay_stops(root):
    for t in "ab":
        fleet_quiet.hold(_ev(t))
    r = _runner()

    async def reengage(ev):
        r.handled.append(ev.text)
        _quiet(root)
    r.adapter.handle_message = reengage
    asyncio.run(r._fleet_quiet_tick())
    assert r.handled == ["a"] and [x["text"] for x in fleet_quiet.load_held()] == ["b"]


def test_held_during_unfinished_replay_queues_behind_backlog(root):
    for t in "ab":
        fleet_quiet.hold(_ev(t))
    fleet_quiet.claim_held()  # an interrupted replay still owns a, b
    assert fleet_quiet.hold(_ev("c"))["first"] is False  # channel already noticed for this backlog
    r = _runner()
    asyncio.run(r._fleet_quiet_tick())
    assert r.handled == ["a", "b", "c"]


def test_disk_failure_falls_back_to_memory(root, monkeypatch):
    r = _runner()
    _quiet(root)

    def boom(ev):
        raise OSError("disk full")
    monkeypatch.setattr(fleet_quiet, "hold", boom)
    # exc_info rendering reads stdlib sources via linecache; keep the log quiet in-test.
    import logging
    import gateway.run_shutdown as rs
    monkeypatch.setattr(rs, "logger", logging.getLogger("fleet-quiet-test-null"))
    logging.getLogger("fleet-quiet-test-null").disabled = True

    async def go():
        await r._fleet_quiet_hold(_ev("m1"))
        await r._fleet_quiet_hold(_ev("m2"))
        (root / fleet_quiet.MARKER_NAME).unlink()
        await r._fleet_quiet_tick()

    asyncio.run(go())
    assert len(r.sent) == 1 and r.handled == ["m1", "m2"]


def test_queue_full_sends_single_notice(root, monkeypatch):
    monkeypatch.setattr(fleet_quiet, "MAX_HELD", 0)
    r = _runner()
    _quiet(root)

    async def go():
        await r._fleet_quiet_hold(_ev("x"))
        await r._fleet_quiet_hold(_ev("y"))

    asyncio.run(go())
    assert len(r.sent) == 1


# ── admission placement (every adapter funnels through _hm_admit_event) ─────────────────────────

def _admit_runner(authorized=True):
    r = object.__new__(GatewayRunner)
    r.config = SimpleNamespace(multiplex_profiles=False)
    r.held = []
    r._startup_restore_in_progress = False

    async def hold(ev):
        r.held.append(ev.text)
    r._fleet_quiet_hold = hold
    r._is_user_authorized_for_source = lambda src: authorized
    return r


@pytest.mark.parametrize("plat", [Platform.TELEGRAM, Platform.SLACK, Platform("photon")])
def test_admission_holds_authorized_inbound_on_every_channel(root, plat, monkeypatch):
    monkeypatch.setattr(run_mod, "_is_slack_ignored_channel", lambda *a, **k: False)
    _quiet(root)
    r = _admit_runner()
    assert asyncio.run(r._hm_admit_event(_ev("hello", plat=plat))) is None
    assert r.held == ["hello"]


def test_admission_does_not_hold_unauthorized(root):
    _quiet(root)
    r = _admit_runner(authorized=False)
    r._get_unauthorized_dm_behavior = lambda *a, **k: None
    reported = []

    async def report(src):
        reported.append(src.chat_id)
    r._hm_report_ignored_dm = report
    r._scale_to_zero_note_real_inbound = lambda: None

    async def hook(ev, src):
        return ev
    r._hm_pre_gateway_dispatch_hook = hook
    ev = _ev("stranger")
    ev.source.user_id = "u-x"
    assert asyncio.run(r._hm_admit_event(ev)) is None
    assert r.held == [] and fleet_quiet.load_held() == [] and reported == ["c1"]


def test_admission_lets_internal_events_through(root):
    _quiet(root)
    r = _admit_runner()
    ev = _ev("internal")
    ev.internal = True
    assert asyncio.run(r._hm_admit_event(ev)) is not None
    assert r.held == []


# ── api_server (Agora rooms, /v1/runs, cron fire) ───────────────────────────────────────────────

def test_api_server_returns_503_fleet_update_quiet(root):
    pytest.importorskip("aiohttp")
    from gateway.platforms.api_server import APIServerAdapter
    a = object.__new__(APIServerAdapter)
    assert a._fleet_quiet_response() is None
    _quiet(root, ttl=30)
    resp = a._draining_response()
    assert resp is not None and resp.status == 503
    body = json.loads(resp.body)
    assert body["error"]["code"] == "fleet_update_quiet"
    assert body["error"]["message"] == fleet_quiet.NOTICE_TEXT
    assert 1 <= int(resp.headers["Retry-After"]) <= 30


# ── background dispatch gates ───────────────────────────────────────────────────────────────────

def test_kanban_dispatch_blocked(root):
    from gateway.kanban_watchers_common import _kanban_dispatch_allowed
    _quiet(root, phase="passed")
    assert _kanban_dispatch_allowed() is False


def test_review_queue_blocked(root):
    from agent import review_idle_queue as q
    _quiet(root)
    assert q._fleet_quiet() is True
    assert q.ReviewIdleQueue()._pop_dispatchable() is None
