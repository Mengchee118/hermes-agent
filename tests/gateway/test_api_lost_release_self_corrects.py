"""A lost API-work release must not leave the gateway BUSY until a restart.

LOCAL PATCH (2026-10-18, t_dfc1ae6e): the busy count was bare ``+= 1`` / ``-= 1`` pairs, so one
skipped release (an exception between the increment and its ``try``, a callback that never ran)
held ``active_agents`` above zero forever; the fleet updater refuses to restart a busy gateway, so
the only way out was a forced restart. Every unit is now owned by a live handle, and a unit whose
owner finished without releasing it stops counting on the next read.
"""
import asyncio
import json

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.control_socket import GatewayControlServer
from gateway.platforms import api_server as api_server_mod
from gateway.platforms import api_server_runs, api_server_work
from gateway.platforms.api_server import APIServerAdapter
from gateway.run_bootstrap import host_status_payload
from gateway.status import flush_runtime_status, write_runtime_status
from tests.gateway.restart_test_helpers import make_restart_runner


def _status_verb_active_agents():
    """The persisted count the control socket's ``status`` verb serves."""
    assert flush_runtime_status()  # the status writer is asynchronous
    response = json.loads(GatewayControlServer().handle_request_line(b'{"verb":"status"}'))
    assert response["ok"], response
    return response["result"].get("active_agents")


def _live_status(runner):
    """What the gateway's live ``status`` verb reports (run_bootstrap.host_status_payload)."""
    return host_status_payload(runner)["active_agents"]


async def _settle():
    for _ in range(3):
        await asyncio.sleep(0)


def _wire(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    write_runtime_status(gateway_state="running", active_agents=0)
    runner, _ = make_restart_runner()
    runner._active_cron_job_count = lambda: 0
    runner._active_deferred_agent_worker_count = lambda: 0
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    runner.adapters[Platform.API_SERVER] = adapter
    return runner, adapter


@pytest.mark.asyncio
async def test_lost_inflight_release_returns_status_to_zero_without_restart(tmp_path, monkeypatch):
    runner, adapter = _wire(tmp_path, monkeypatch)
    started, finish = asyncio.Event(), asyncio.Event()

    async def parked_worker(_loop, _fn, *, on_finished=None):
        started.set()
        await finish.wait()
        return {"final_response": "ok"}, {}

    monkeypatch.setattr(api_server_runs, "_submit_api_worker", parked_worker)
    real_release = api_server_work.release
    dropped = []

    def drop_inflight_release(adapter_, slot):
        # Reproduce a lost release: the turn's own decrement never happens.
        if slot and slot.get("kind") == "inflight" and not dropped:
            dropped.append(slot)
            return False
        return real_release(adapter_, slot)

    monkeypatch.setattr(api_server_work, "release", drop_inflight_release)
    turn = asyncio.create_task(adapter._run_agent(
        user_message="hello", conversation_history=[], session_id="s1"))
    await asyncio.wait_for(started.wait(), 2)
    assert _live_status(runner) == 1  # a live turn is never reclaimed
    finish.set()
    await asyncio.wait_for(turn, 2)
    assert dropped and adapter._inflight_agent_runs == 1  # the release really was lost

    # Next status read: the lost unit no longer counts, from the read alone.
    assert _live_status(runner) == 0
    # The owning loop then releases it for real and republishes the persisted count.
    await _settle()
    assert adapter._inflight_agent_runs == 0
    assert _status_verb_active_agents() == 0
    # A late legitimate release after the reclaim is a no-op (never double-counted).
    assert real_release(adapter, dropped[0]) is False
    assert adapter._inflight_agent_runs == 0


@pytest.mark.asyncio
async def test_failure_after_admission_still_releases_the_turn(tmp_path, monkeypatch):
    """The reviewer's path: the in-flight increment ran before the ``try``, so a raise from
    ``_activate_admitted_request`` skipped the ``finally`` and held the count at 1 forever."""
    runner, adapter = _wire(tmp_path, monkeypatch)

    def boom():
        raise RuntimeError("activation failed")

    adapter._activate_admitted_request = boom
    with pytest.raises(RuntimeError, match="activation failed"):
        await adapter._run_agent(user_message="hello", conversation_history=[], session_id="s1")
    assert adapter._inflight_agent_runs == 0
    assert _live_status(runner) == 0
    assert _status_verb_active_agents() == 0


@pytest.mark.asyncio
async def test_detached_reservation_stays_busy_until_its_fire_task_ends(tmp_path, monkeypatch):
    """False SAFE is the dangerous failure: work handed to a detached task must stay counted
    after the admitting handler returns, and clear once that task finishes."""
    runner, adapter = _wire(tmp_path, monkeypatch)
    finish = asyncio.Event()

    async def handler():
        with api_server_mod._reserve_pending_api_work(adapter) as reservation:
            fire = asyncio.create_task(finish.wait())
            reservation["detached"] = True
            reservation["owner"] = fire
            fire.add_done_callback(
                lambda _t: api_server_mod._release_pending_api_work(adapter, reservation))
            return fire

    fire = await handler()
    await _settle()
    assert _live_status(runner) == 1
    finish.set()
    await fire
    await _settle()
    assert _live_status(runner) == 0
    assert adapter._pending_agent_requests == 0


@pytest.mark.asyncio
async def test_lost_pending_reservation_self_corrects(tmp_path, monkeypatch):
    runner, adapter = _wire(tmp_path, monkeypatch)

    async def admitted_then_lost():
        api_server_work.acquire(adapter, "pending")  # nobody will release this

    await asyncio.create_task(admitted_then_lost())
    assert adapter._pending_agent_requests == 1
    assert _live_status(runner) == 0
    await _settle()
    assert adapter._pending_agent_requests == 0
    assert _status_verb_active_agents() == 0


@pytest.mark.asyncio
async def test_native_turn_during_agora_turn_does_not_bake_a_stale_count(tmp_path, monkeypatch):
    """The 2026-09-25 report: an Agora (API-server) turn starts, a Telegram turn claims and
    releases while it runs, the Agora turn ends -> the status verb must read 0, not 1."""
    runner, adapter = _wire(tmp_path, monkeypatch)
    started, finish = asyncio.Event(), asyncio.Event()

    async def parked_worker(_loop, _fn, *, on_finished=None):
        started.set()
        await finish.wait()
        return {"final_response": "ok"}, {}

    monkeypatch.setattr(api_server_runs, "_submit_api_worker", parked_worker)
    agora = asyncio.create_task(adapter._run_agent(
        user_message="hello", conversation_history=[], session_id="agora-1"))
    await asyncio.wait_for(started.wait(), 2)

    # Telegram turn claims (native boundary persists, baking the Agora turn in) ...
    runner._running_agents["agent:main:telegram:dm:1"] = object()
    runner._persist_active_agents()
    assert _status_verb_active_agents() == 2
    # ... and releases while the Agora turn is still running.
    del runner._running_agents["agent:main:telegram:dm:1"]
    runner._persist_active_agents()
    assert _status_verb_active_agents() == 1

    finish.set()
    await asyncio.wait_for(agora, 2)
    await _settle()
    assert _status_verb_active_agents() == 0
