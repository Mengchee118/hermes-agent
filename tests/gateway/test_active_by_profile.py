"""The shared gateway attributes the host total without multiplying it by served profiles."""
from collections import Counter
from unittest.mock import Mock

import pytest

from gateway.run import GatewayRunner
from gateway.config import Platform
from gateway.platforms.api_server import APIServerAdapter


def test_one_native_turn_and_idle_peer(monkeypatch):
    import cron.scheduler as scheduler
    runner = object.__new__(GatewayRunner)
    runner.served_profile_names = lambda: ['default', 'athena', 'iris']
    runner._running_agent_items = lambda: [('agent:athena:agora:room:r', object())]
    runner._active_work_count = lambda: 1
    runner.adapters = {}
    monkeypatch.setattr(scheduler, 'get_running_job_details', lambda: [])
    assert runner._active_by_profile() == {'default': 0, 'athena': 1, 'iris': 0}


def test_api_work_owner_and_total(monkeypatch):
    import cron.scheduler as scheduler
    runner = object.__new__(GatewayRunner)
    runner.served_profile_names = lambda: ['default', 'athena', 'iris']
    runner._running_agent_items = lambda: []
    runner._active_work_count = lambda: 1
    adapter = Mock()
    adapter.active_agent_work_by_profile.return_value = {'athena': 1}
    runner.adapters = {Platform.API_SERVER: adapter}
    monkeypatch.setattr(scheduler, 'get_running_job_details', lambda: [])
    assert runner._active_by_profile() == {'default': 0, 'athena': 1, 'iris': 0}
    runner._active_work_count = lambda: 2
    with pytest.raises(ValueError, match='host total'):
        runner._active_by_profile()


def test_api_adapter_pending_inflight_and_run_owners():
    adapter = object.__new__(APIServerAdapter)
    adapter._pending_by_profile = Counter({'athena': 1})
    adapter._inflight_by_profile = Counter({'iris': 1})
    adapter._orphaned_by_profile = Counter()
    adapter._active_run_profiles = {'run-1': 'athena'}
    adapter._active_run_tasks = {'run-1': Mock(**{'done.return_value': False})}
    adapter.active_agent_work_count = lambda: 3
    assert adapter.active_agent_work_by_profile() == {'athena': 2, 'iris': 1}
    adapter._active_run_profiles.clear()
    with pytest.raises(ValueError, match='owning profile'):
        adapter.active_agent_work_by_profile()


def test_cron_and_deferred_worker_profile_owners(monkeypatch, tmp_path):
    import cron.scheduler as scheduler
    import hermes_constants
    root = tmp_path / 'hermes'
    runner = object.__new__(GatewayRunner)
    runner.served_profile_names = lambda: ['default', 'athena', 'iris']
    runner._running_agent_items = lambda: []
    runner._active_work_count = lambda: 2
    runner.adapters = {}
    worker = Mock(**{'done.return_value': False})
    runner._deferred_agent_workers = {worker: object()}
    runner._deferred_worker_profiles = {worker: 'iris'}
    monkeypatch.setattr(hermes_constants, 'get_hermes_home', lambda: root)
    monkeypatch.setattr(scheduler, 'get_running_job_details', lambda: [
        {'job_id': 'one', 'home': str(root / 'profiles' / 'athena')}])
    assert runner._active_by_profile() == {'default': 0, 'athena': 1, 'iris': 1}


def test_post_reply_review_is_profile_work_and_releases_on_error(monkeypatch, tmp_path):
    """Review remains BUSY after the foreground agent slot has been released."""
    import hermes_constants
    import cron.scheduler as scheduler
    from agent.background_review import (
        active_background_reviews_by_home, finish_background_review_run,
        prepare_background_review_run,
    )
    root = tmp_path / 'hermes'
    home = root / 'profiles' / 'athena'
    agent = Mock()
    agent._background_review_run = None
    agent._background_review_lock = __import__('threading').Lock()
    monkeypatch.setattr(hermes_constants, 'get_hermes_home', lambda: home)
    run = prepare_background_review_run(agent)
    assert run is not None
    monkeypatch.setattr(hermes_constants, 'get_hermes_home', lambda: root)
    runner = object.__new__(GatewayRunner)
    runner.served_profile_names = lambda: ['default', 'athena', 'iris']
    runner._running_agent_items = lambda: []
    runner._running_agent_count = lambda: 0
    runner._active_cron_job_count = lambda: 0
    runner._active_api_run_count = lambda: 0
    runner._active_deferred_agent_worker_count = lambda: 0
    runner.adapters = {}
    monkeypatch.setattr(scheduler, 'get_running_job_details', lambda: [])
    try:
        assert runner._active_work_count() == 1
        assert runner._active_by_profile() == {'default': 0, 'athena': 1, 'iris': 0}
        assert runner._drain_work_counts() == (0, 0, 0, 0, 1)
    finally:
        # The same finally path must run if the review's tool/provider raises.
        finish_background_review_run(agent, run)
    finish_background_review_run(agent, run)  # repeated cleanup is harmless
    assert active_background_reviews_by_home() == {}
    assert runner._active_work_count() == 0
    assert runner._active_by_profile() == {'default': 0, 'athena': 0, 'iris': 0}
