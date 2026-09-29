"""API turns must refresh the same persisted work count native turns use."""
import asyncio
import json
import threading
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.control_socket import GatewayControlServer
from gateway.platforms import api_server_runs
from gateway.platforms.api_server import APIServerAdapter
from gateway.status import write_runtime_status
from tests.gateway.restart_test_helpers import make_restart_runner


async def _await_status_count(expected):
    for _ in range(100):
        # Exercise the same status verb the fleet UI queries, not the writer alone.
        server = GatewayControlServer()
        response = json.loads(server.handle_request_line(b'{"verb":"status"}'))
        assert response["ok"], response
        record = response["result"]
        if record.get("active_agents") == expected:
            return record
        await asyncio.sleep(0.01)
    pytest.fail(f"status did not reach {expected}: {record}")


@pytest.mark.asyncio
async def test_api_turn_interleaved_with_native_turn_returns_status_to_idle(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    write_runtime_status(gateway_state="running", active_agents=0)
    runner, _ = make_restart_runner()
    runner._active_cron_job_count = lambda: 0
    runner._active_deferred_agent_worker_count = lambda: 0
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    runner.adapters[Platform.API_SERVER] = adapter
    started, release = asyncio.Event(), asyncio.Event()

    async def parked_worker(_loop, _fn, *, on_finished=None):
        started.set()
        await release.wait()
        return {"final_response": "ok"}, {}

    monkeypatch.setattr(api_server_runs, "_submit_api_worker", parked_worker)
    turn = asyncio.create_task(adapter._run_agent(
        user_message="hello", conversation_history=[], session_id="s1"))
    await asyncio.wait_for(started.wait(), 2)
    await _await_status_count(1)
    runner._running_agents["telegram"] = object()
    runner._persist_active_agents()
    await _await_status_count(2)
    runner._running_agents.pop("telegram")
    runner._persist_active_agents()
    await _await_status_count(1)
    release.set()
    await asyncio.wait_for(turn, 2)
    await _await_status_count(0)
    assert (await _await_status_count(0))["gateway_state"] == "running"


@pytest.mark.asyncio
async def test_api_turn_exception_still_publishes_idle(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner, _ = make_restart_runner()
    runner._active_cron_job_count = lambda: 0
    runner._active_deferred_agent_worker_count = lambda: 0
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    runner.adapters[Platform.API_SERVER] = adapter

    async def fail_worker(_loop, _fn, *, on_finished=None):
        await _await_status_count(1)
        raise RuntimeError("worker failed")

    monkeypatch.setattr(api_server_runs, "_submit_api_worker", fail_worker)
    with pytest.raises(RuntimeError, match="worker failed"):
        await adapter._run_agent(user_message="hello", conversation_history=[], session_id="s1")
    await _await_status_count(0)


@pytest.mark.asyncio
async def test_structured_run_publishes_start_and_completion(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner, _ = make_restart_runner()
    runner._active_cron_job_count = lambda: 0
    runner._active_deferred_agent_worker_count = lambda: 0
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    runner.adapters[Platform.API_SERVER] = adapter
    started, release = asyncio.Event(), asyncio.Event()

    async def parked_run(_adapter, _launch, *, _api_server):
        started.set()
        await release.wait()

    app = web.Application()
    app.router.add_post("/v1/runs", adapter._handle_runs)
    with patch.object(api_server_runs, "_execute_run", parked_run):
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/v1/runs", json={"input": "hello"})
            assert response.status == 202
            await asyncio.wait_for(started.wait(), 2)
            await _await_status_count(1)
            release.set()
            await asyncio.gather(*adapter._active_run_tasks.values())
            await _await_status_count(0)


@pytest.mark.asyncio
async def test_cancelled_api_handler_keeps_worker_busy_until_exit(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner, _ = make_restart_runner()
    runner._active_cron_job_count = lambda: 0
    runner._active_deferred_agent_worker_count = lambda: 0
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    runner.adapters[Platform.API_SERVER] = adapter
    loop = asyncio.get_running_loop()
    entered, release = asyncio.Event(), threading.Event()
    agent = MagicMock()
    agent.session_id = None
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0

    def parked_turn(**_kwargs):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return {"final_response": "done", "messages": [], "api_calls": 0, "tools": []}

    agent.run_conversation.side_effect = parked_turn
    try:
        with patch.object(adapter, "_create_agent", return_value=agent):
            turn = asyncio.create_task(adapter._run_agent(
                user_message="hello", conversation_history=[], session_id="s1"))
            await asyncio.wait_for(entered.wait(), 2)
            await _await_status_count(1)
            turn.cancel()
            with pytest.raises(asyncio.CancelledError):
                await turn
            assert adapter._orphaned_api_workers == 1
            await _await_status_count(1)
    finally:
        release.set()
    for _ in range(100):
        if adapter._orphaned_api_workers == 0:
            break
        await asyncio.sleep(0.01)
    assert adapter._orphaned_api_workers == 0
    await _await_status_count(0)
