"""Regression tests for #60432.

``/update`` (and other gateway shutdown paths) must drain in-flight cron jobs
before ``process_registry.kill_all()`` runs in final-cleanup.  Cron work runs on
a thread-pool worker and is tracked in ``cron.scheduler._running_job_ids``, not
in ``GatewayRunner._running_agents`` — so a zero-agent drain must still wait
for cron to finish (or time out and take the interrupt/kill path).
"""
import asyncio
from unittest.mock import patch

import pytest

from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.mark.asyncio
async def test_drain_active_agents_waits_for_in_flight_cron_jobs():
    runner, _adapter = make_restart_runner()
    runner._running_agents = {}

    observed = asyncio.Event()
    released = asyncio.Event()

    def _cron_in_flight():
        observed.set()
        return frozenset() if released.is_set() else frozenset({"job-1"})

    with patch("cron.scheduler.get_running_job_ids", side_effect=_cron_in_flight):
        drain = asyncio.create_task(runner._drain_active_agents(10.0))
        try:
            await asyncio.wait_for(observed.wait(), timeout=5.0)
            assert not drain.done(), "drain returned while cron still owned its work"
            released.set()
            _snapshot, timed_out = await asyncio.wait_for(drain, timeout=5.0)
        finally:
            released.set()
            if not drain.done():
                drain.cancel()
            await asyncio.gather(drain, return_exceptions=True)

    assert timed_out is False
    assert _snapshot == {}


@pytest.mark.asyncio
async def test_drain_waits_for_post_reply_review(monkeypatch, tmp_path):
    from agent.background_review import prepare_background_review_run, finish_background_review_run
    import hermes_constants
    from types import SimpleNamespace
    from threading import Lock

    monkeypatch.setattr(hermes_constants, 'get_hermes_home', lambda: tmp_path)
    agent = SimpleNamespace(_background_review_run=None, _background_review_lock=Lock())
    run = prepare_background_review_run(agent)
    assert run is not None
    runner, _adapter = make_restart_runner()
    runner._running_agents = {}
    drain = asyncio.create_task(runner._drain_active_agents(5.0, 5.0))
    try:
        await asyncio.sleep(0.2)
        assert not drain.done(), 'drain must not pass while a review can still write a skill'
        assert runner._awaitable_work_count() >= 1
    finally:
        finish_background_review_run(agent, run)
    _snapshot, timed_out = await asyncio.wait_for(drain, timeout=5.0)
    assert _snapshot == {}
    assert timed_out is False


