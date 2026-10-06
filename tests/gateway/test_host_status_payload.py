"""LOCAL PATCH: ``status`` reports the live host-level in-flight count (post-reply reviews included)."""
from types import SimpleNamespace

from gateway.run_bootstrap import host_status_payload
from gateway.status import flush_runtime_status, write_runtime_status


def _persist_idle(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    write_runtime_status(gateway_state="running", active_agents=0)
    flush_runtime_status()


def test_status_reports_live_work_the_file_has_not_seen(tmp_path, monkeypatch):
    _persist_idle(tmp_path, monkeypatch)
    payload = host_status_payload(SimpleNamespace(_active_work_count=lambda: 2))
    assert payload["active_agents"] == 2


def test_status_counts_a_post_reply_review(tmp_path, monkeypatch):
    import hermes_constants
    from threading import Lock
    from agent.background_review import prepare_background_review_run, finish_background_review_run
    from tests.gateway.restart_test_helpers import make_restart_runner

    _persist_idle(tmp_path, monkeypatch)
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    runner, _adapter = make_restart_runner()
    runner._running_agents = {}
    assert host_status_payload(runner)["active_agents"] == 0
    agent = SimpleNamespace(_background_review_run=None, _background_review_lock=Lock())
    run = prepare_background_review_run(agent)
    try:
        assert host_status_payload(runner)["active_agents"] >= 1
    finally:
        finish_background_review_run(agent, run)
    assert host_status_payload(runner)["active_agents"] == 0


def test_status_falls_back_to_persisted_count_when_live_count_fails(tmp_path, monkeypatch):
    _persist_idle(tmp_path, monkeypatch)

    def boom():
        raise RuntimeError("no count")

    payload = host_status_payload(SimpleNamespace(_active_work_count=boom))
    assert payload["active_agents"] == 0
