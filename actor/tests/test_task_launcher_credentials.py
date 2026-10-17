import asyncio
import base64
import importlib.util
import json
import os
import time
from pathlib import Path

import pytest


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec = importlib.util.spec_from_file_location(
        "run_lean_advice_task_credentials", Path("scripts/run_lean_advice_task.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_explicit_file_overrides_stale_environment_without_output(tmp_path, monkeypatch, capsys):
    mod = module(monkeypatch)
    # Import-time tau/LiteLLM diagnostics are unrelated to credential handling.
    capsys.readouterr()
    monkeypatch.setenv("OPENAI_API_KEY", "stale-key")
    key = "sk-proj-" + "TEST_FIXTURE_NOT_A_REAL_KEY" * 2
    path = tmp_path / "api-key.txt"
    path.write_text(key + "\n")
    mod.configure_api_key(path)
    assert os.environ["OPENAI_API_KEY"] == key
    captured = capsys.readouterr()
    assert not captured.out
    assert not captured.err


@pytest.mark.parametrize(
    "value", ["", "not-a-key", "sk-short", "sk-proj-" + "x" * 600, "sk-proj-" + "a b" * 10]
)
def test_invalid_file_fails_without_using_or_echoing_ambient_key(tmp_path, monkeypatch, value):
    mod = module(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "stale-key")
    path = tmp_path / "api-key.txt"
    path.write_text(value)
    with pytest.raises(ValueError, match="single OpenAI API key") as caught:
        mod.configure_api_key(path)
    assert os.environ["OPENAI_API_KEY"] == "stale-key"
    assert "stale-key" not in str(caught.value)
    if value:
        assert value not in str(caught.value)


def test_missing_file_has_safe_error_and_never_falls_back(tmp_path, monkeypatch):
    mod = module(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "stale-key")
    with pytest.raises(ValueError, match="Cannot read"):
        mod.configure_api_key(tmp_path / "missing")
    assert os.environ["OPENAI_API_KEY"] == "stale-key"


def test_subscription_uses_explicit_file_without_default_fallback(tmp_path, monkeypatch):
    mod = module(monkeypatch)
    claims = base64.urlsafe_b64encode(
        json.dumps({"exp": time.time() + 3600}).encode()
    ).decode().rstrip("=")
    access = "fixture." + claims + ".fixture"
    path = tmp_path / "selected.json"
    path.write_text(json.dumps({"tokens": {
        "access_token": access, "refresh_token": "fixture-refresh",
        "account_id": "fixture-account",
    }}))
    monkeypatch.setenv("CAYU_TAU_CODEX_AUTH_FILE", str(path))
    credentials = asyncio.run(mod._SubscriptionAuth().credentials())
    assert credentials.access_token == access
    assert credentials.account_id == "fixture-account"
    path.unlink()
    with pytest.raises(FileNotFoundError):
        asyncio.run(mod._SubscriptionAuth().credentials())


def test_launcher_installs_recovering_subscription_provider(monkeypatch):
    from cayu_tau3_banking.subscription_recovery import RecoveringSubscriptionProvider
    mod = module(monkeypatch)
    monkeypatch.setattr(mod.cli, "build_provider", mod.cli.build_provider)
    mod.configure_actor_provider({"cli_arguments": ["--provider", "subscription"]})
    provider = mod.cli.build_provider(provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=60)
    assert isinstance(provider, RecoveringSubscriptionProvider)
    asyncio.run(provider.aclose())
