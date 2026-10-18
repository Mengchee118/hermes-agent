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


def test_every_work_kind_is_attributed_and_sums_to_host_total(monkeypatch, tmp_path):
    """LOCAL PATCH (2026-10-18, t_dfc1ae6e): each kind in ``_active_work_count`` must have a
    per-profile source. Managed workers (Kanban / agent.managed_worker turns), unmapped executing
    claims and owner mutations were counted in the host total only, so one live managed turn made
    attribution raise and every served profile reported the host total (all-7 BUSY)."""
    import threading
    from cron import scheduler
    import hermes_constants
    from types import SimpleNamespace
    from agent.background_review import active_background_reviews_by_home
    import agent.background_review as br

    root = tmp_path / 'hermes'
    home = lambda name: root if name == 'default' else root / 'profiles' / name
    names = ['default', 'athena', 'iris', 'zeus', 'argus', 'star', 'venus', 'pepper']
    monkeypatch.setattr(hermes_constants, 'get_hermes_home', lambda: root)

    def open_worker():
        return SimpleNamespace(closed=threading.Event())

    def live_task():
        return Mock(**{'done.return_value': False})

    def authority(name, *, managed=0, unmapped=0, mutations=0):
        sessions = {f'{name}-claim-{i}': SimpleNamespace(
            route=f'route-{name}-{i}', task=live_task(),
            event_stream=SimpleNamespace(execution={'execution_generation': 1}))
            for i in range(unmapped)}
        return SimpleNamespace(
            profile_id=str(home(name)),
            _managed_workers={f'{name}-w{i}': open_worker() for i in range(managed)},
            sessions=sessions,
            _mutation_tasks={live_task() for _ in range(mutations)},
            _bot_mailbox_operations=set())

    runner = object.__new__(GatewayRunner)
    runner.served_profile_names = lambda: names
    # kind 1: native running turn (athena)
    runner._running_agents = {'agent:athena:agora:room:r': object()}
    runner._running_agent_items = lambda: list(runner._running_agents.items())
    runner._running_agent_count = lambda: len(runner._running_agents)
    # kinds 2-4: managed worker (zeus = this Kanban turn), unmapped claim (iris), mutation (argus)
    authorities = [authority('default'), authority('athena'), authority('iris', unmapped=1),
                   authority('zeus', managed=1), authority('argus', mutations=1),
                   authority('star'), authority('venus'), authority('pepper')]
    runner.session_authorities = authorities
    # kind 5: cron (star)
    monkeypatch.setattr(scheduler, 'get_running_job_details',
                        lambda: [{'job_id': 'j', 'home': str(home('star'))}])
    monkeypatch.setattr(scheduler, 'get_running_job_ids', lambda: ['j'])
    # kind 6: api_server (venus)
    adapter = Mock()
    adapter.active_agent_work_by_profile.return_value = {'venus': 1}
    adapter.active_agent_work_count.return_value = 1
    runner.adapters = {Platform.API_SERVER: adapter}
    # kind 7: deferred worker (default)
    worker = live_task()
    runner._deferred_agent_workers = {worker: object()}
    runner._deferred_worker_profiles = {worker: 'default'}
    runner._active_deferred_agent_worker_count = lambda: 1
    # kind 8: post-reply background review (pepper)
    monkeypatch.setattr(br, 'active_background_reviews_by_home', lambda: {str(home('pepper')): 1})

    total = runner._active_work_count()
    assert total == 8
    by_profile = runner._active_by_profile()
    assert sum(by_profile.values()) == total
    assert by_profile == {name: 1 for name in names}

    # Only the managed Kanban worker live (the 07:49 boot shape): ONE profile busy, six idle.
    runner._running_agents = {}
    runner.session_authorities = [authority(n, managed=int(n == 'zeus')) for n in names]
    monkeypatch.setattr(scheduler, 'get_running_job_details', list)
    monkeypatch.setattr(scheduler, 'get_running_job_ids', list)
    adapter.active_agent_work_by_profile.return_value = {}
    adapter.active_agent_work_count.return_value = 0
    runner._deferred_agent_workers = {}
    runner._active_deferred_agent_worker_count = lambda: 0
    monkeypatch.setattr(br, 'active_background_reviews_by_home', dict)
    assert runner._active_work_count() == 1
    assert runner._active_by_profile() == {n: int(n == 'zeus') for n in names}

    # A kind with no per-profile source still refuses (host-wide BUSY stays the safe fallback).
    runner._active_work_count = lambda: 2
    with pytest.raises(ValueError, match='host total'):
        runner._active_by_profile()
