import gzip
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec = importlib.util.spec_from_file_location("official_campaign_test", "scripts/run_official_campaign.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_keeps_scored_failures_and_original_unscored_attempts(campaign, tmp_path):
    directory = tmp_path / "official"
    directory.mkdir()
    payload = {"simulations": [
        {"reward_info": {"reward": 1}}, {"reward_info": {"reward": 0}},
        {"reward_info": None, "termination_reason": "infrastructure_error"},
    ]}
    data = json.dumps(payload).encode()
    (directory / "results.json").write_bytes(data)
    campaign.preserve_checkpoint(tmp_path)
    record = next((tmp_path / "attempt-history").glob("*.gz"))
    assert gzip.decompress(record.read_bytes()) == data
    assert (directory / "results.json").read_bytes() == data
    assert campaign.summary(tmp_path) == {
        "expected": 388, "finished": 3, "passed": 1, "scored_failures": 1, "unscored": 1,
    }


def test_checkpoint_being_written_does_not_crash_supervisor(campaign, tmp_path):
    (tmp_path / "official").mkdir()
    (tmp_path / "official/results.json").write_text('{"simulations":')
    assert campaign.summary(tmp_path)["snapshot_pending"]


@pytest.mark.parametrize("status,pauses", [(401, True), (403, True), (429, False), (500, False), (503, False)])
def test_http_boundary_without_json_error_pauses_only_auth(campaign, tmp_path, status, pauses):
    directory = tmp_path / "provider-diagnostics" / "provider-test"
    directory.mkdir(parents=True)
    path = directory / "request.error.json"
    path.write_text(json.dumps({"schema": "provider_boundary", "error": {},
                                "http_status_code": status, "error_status_conflict": False}))
    assert bool(campaign.shared_failure(tmp_path, 0)) is pauses
    assert campaign.shared_failure(tmp_path, path.stat().st_mtime + 1) is None


def test_conflicting_or_untyped_status_does_not_infer_auth_outage(campaign):
    assert not campaign.explicit_shared_outage({"http_status_code": 401, "error_status_conflict": True})
    assert not campaign.explicit_shared_outage({"message": "request failed 401"})
    assert not campaign.explicit_shared_outage({"http_status_code": "401"})
    assert campaign.explicit_shared_outage({"error": {"status_code": 403}})


def test_only_new_shared_outages_pause(campaign, tmp_path):
    directory = tmp_path / "provider-diagnostics/provider-test"
    directory.mkdir(parents=True)
    path = directory / "request.error.json"
    path.write_text(json.dumps({"error": {"type": "service_unavailable_error"}}))
    assert campaign.shared_failure(tmp_path, 0) is None
    path.write_text(json.dumps({"error": {"type": "usage_limit_reached"}}))
    assert campaign.shared_failure(tmp_path, 0) == "provider_auth_or_quota"
    assert campaign.shared_failure(tmp_path, path.stat().st_mtime + 1) is None


@pytest.mark.parametrize("identity,pauses", [
    ("RateLimitError", False), ("rate_limit_exceeded", False),
    ("service_unavailable_error", False), ("insufficient_quota", True),
    ("AuthenticationError", True), ("account_deactivated", True),
])
def test_transient_throttling_does_not_latch_official_campaign(campaign, tmp_path, identity, pauses):
    (tmp_path / "official").mkdir()
    (tmp_path / "official/results.json").write_text(json.dumps({"simulations": [{
        "timestamp": "2026-09-26T00:00:00+00:00", "termination_reason": "infrastructure_error",
        "info": {"error_type": identity},
    }]}))
    assert bool(campaign.shared_failure(tmp_path, 0)) is pauses
    (tmp_path / "official/results.json").write_text('{"simulations": []}')
    directory = tmp_path / "provider-diagnostics/provider-test"
    directory.mkdir(parents=True)
    (directory / "request.error.json").write_text(json.dumps({"error": {"code": identity}}))
    assert bool(campaign.shared_failure(tmp_path, 0)) is pauses


@pytest.mark.parametrize("trials", [1, 4])
def test_full_domain_uses_upstream_resume_and_no_task_filters(campaign, tmp_path, monkeypatch, trials):
    from types import SimpleNamespace

    from tau2.data_model.simulation import TextRunConfig
    received = []
    (tmp_path / "protocol.json").write_text(json.dumps({"trials": trials}))
    config = TextRunConfig(domain="banking_knowledge", num_trials=trials, seed=300)
    cli = SimpleNamespace(build_memory_corpus=None, run_domain=lambda cfg: received.append(cfg))
    cli.main = lambda: cli.run_domain(config)
    monkeypatch.setattr(campaign, "verify", lambda root: None)
    monkeypatch.setattr(campaign, "install_runtime", lambda root, key: (cli, {
        "cli_arguments": ["--num-trials", "4", "--max-concurrency", "8"],
        "tau_repository": str(tmp_path),
    }))
    monkeypatch.setattr(campaign.sys, "argv", ["test-campaign"])
    campaign.worker(tmp_path, tmp_path / "unused-key")
    assert len(received) == 1
    assert received[0].auto_resume
    assert received[0].num_trials == trials
    assert received[0].seed == 300
    assert received[0].task_ids is None and received[0].num_tasks is None
    assert "--task-ids" not in campaign.sys.argv


@pytest.mark.parametrize("trials", [1, 4])
def test_registered_full_domain_count(campaign, tmp_path, trials):
    (tmp_path / "protocol.json").write_text(json.dumps({"trials": trials, "expected_trajectories": 97 * trials}))
    assert campaign.summary(tmp_path)["expected"] == 97 * trials


@pytest.mark.parametrize("protocol", [{"trials": 2}, {"trials": True}, {"trials": 1, "expected_trajectories": 2}])
def test_rejects_inconsistent_campaign_size(campaign, tmp_path, protocol):
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError):
        campaign.expected_trials(tmp_path)
