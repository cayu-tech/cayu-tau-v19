"""Run one preregistered real task on frozen lean-advice sources, without retries."""

import argparse
import base64
import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_section_embeddings import CachedEmbeddings  # noqa: E402

from cayu_tau3_banking import cli  # noqa: E402
from cayu_tau3_banking.failure_categories import classify_failure_info  # noqa: E402
from cayu_tau3_banking.retrieval_eval import safe_error_diagnostic  # noqa: E402
from cayu_tau3_banking.screen_diagnostics import (  # noqa: E402
    private_tool_diagnostics,
    termination_diagnostics,
)


class _SubscriptionAuth:
    """Read the user's existing Codex login without copying or logging tokens."""

    async def credentials(self):
        from cayu.providers.openai_subscription import OpenAISubscriptionCredentials

        selected = os.environ.get("CAYU_TAU_CODEX_AUTH_FILE")
        auth_path = Path(selected) if selected else Path.home() / ".codex" / "auth.json"
        payload = json.loads(auth_path.read_text())
        tokens = payload["tokens"]
        access = tokens["access_token"]
        encoded = access.split(".", 2)[1]
        claims = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        expires_at = float(claims["exp"])
        if expires_at <= time.time() + 180:
            raise RuntimeError("Codex subscription access token expires too soon")
        return OpenAISubscriptionCredentials(
            access_token=access,
            refresh_token=tokens["refresh_token"],
            expires_at=expires_at,
            account_id=tokens["account_id"],
        )


def configure_actor_provider(protocol):
    """Route subscription runs to the explicitly selected Codex credential file."""
    arguments = protocol["cli_arguments"]
    if arguments[arguments.index("--provider") + 1] != "subscription":
        return
    from cayu.providers.deadlines import ProviderStreamDeadlines

    from cayu_tau3_banking.subscription_recovery import RecoveringSubscriptionProvider

    def subscription_provider(*, provider, codex_lb_base_url, semantic_progress_timeout_seconds):
        if provider != "subscription":
            raise ValueError("Subscription diagnostic cannot substitute another provider")
        return RecoveringSubscriptionProvider(
            auth=_SubscriptionAuth(),
            stream_deadlines=ProviderStreamDeadlines(
                semantic_progress_timeout_s=semantic_progress_timeout_seconds
            ),
        )

    cli.build_provider = subscription_provider


def save(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(value, stream, indent=2)


def verify():
    manifest = json.loads((ROOT / "source-manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Frozen source changed")
    if not Path(cli.__file__).resolve().is_relative_to(ROOT / "src"):
        raise RuntimeError("Actor is not imported from frozen sources")


def configure_api_key(path: Path) -> None:
    """Explicit file selection overrides ambient credentials without logging them."""
    try:
        api_key = path.read_text().strip()
    except (OSError, UnicodeError):
        raise ValueError("Cannot read the supplied API-key file") from None
    if (
        not api_key.startswith("sk-")
        or not 20 <= len(api_key) <= 512
        or not api_key.isascii()
        or any(c.isspace() for c in api_key)
    ):
        raise ValueError("Expected a single OpenAI API key in the supplied file")
    os.environ["OPENAI_API_KEY"] = api_key


def main():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=protocol["task_ids"], required=True)
    parser.add_argument("--openai-api-key-file", type=Path, required=True)
    args = parser.parse_args()
    verify()
    configure_api_key(args.openai_api_key_file)
    output = ROOT / args.task
    output.mkdir(mode=0o700)
    save(
        output / "started.json",
        {
            "task": args.task,
            "started_unix": time.time(),
            "pid": os.getpid(),
            "actor_source": str(Path(cli.__file__).resolve()),
            "whole_task_retries": 0,
            "platform_credential_source": "explicit_file_overrides_environment",
            "platform_credential_file": str(args.openai_api_key_file.resolve()),
        },
    )
    original_build = cli.build_memory_corpus

    configure_actor_provider(protocol)

    async def cached_corpus(documents, **kwargs):
        cache_root = Path(protocol["corpus_embedding_cache"])
        cache = CachedEmbeddings(kwargs["embedding_provider"])
        manifest = json.loads((cache_root / "manifest.json").read_text())
        cache.restore(cache_root, manifest["corpus_sha256"])
        kwargs["embedding_provider"] = cache
        return await original_build(documents, **kwargs)

    cli.build_memory_corpus = cached_corpus
    original_run = cli.run_domain

    def classify_simulation(sim):
        if sim.termination_reason.value != "infrastructure_error":
            for message in sim.messages or []:
                raw = getattr(message, "raw_data", None)
                if isinstance(raw, dict) and raw.get("agent_termination") in {
                    "planning_stall", "admission_stall", "runtime_limit", "agent_time_budget"
                }:
                    return raw["agent_termination"]
            return None
        info = dict(sim.info or {})
        diagnostics = sorted((output / "provider-diagnostics").glob("*/*.error.json"), key=lambda path: path.stat().st_mtime_ns)
        if diagnostics:
            # The provider adapter writes a typed envelope separately from Tau's
            # generic terminal error. Attach the latest envelope for category
            # projection without copying its message into the assessment.
            info["provider_error"] = json.loads(diagnostics[-1].read_text())
        return classify_failure_info(info)

    def record(config):
        result = original_run(config)
        # This launcher intentionally runs one task/trial per isolated process.
        if len(result.simulations) != 1:
            raise RuntimeError("Screen diagnostics require one isolated simulation")
        private_diagnostics = private_tool_diagnostics(output / "sessions")
        rows = [
            {
                "task": sim.task_id,
                "reward": None if sim.reward_info is None else sim.reward_info.reward,
                "termination": sim.termination_reason.value,
                **termination_diagnostics(sim, classify_simulation(sim)),
                **private_diagnostics,
                "duration_seconds": sim.duration,
                "tool_errors": sum(bool(getattr(m, "error", False)) for m in sim.messages or []),
                "public_tool_errors": sum(bool(getattr(m, "error", False)) for m in sim.messages or []),
                "user_cost": sim.user_cost,
                "agent_cost": sim.agent_cost,
            }
            for sim in result.simulations
        ]
        verify()
        save(output / "assessment.json", {"rows": rows, "source_unchanged": True})
        if not rows or any(r["termination"] == "infrastructure_error" for r in rows):
            raise RuntimeError("Task returned an infrastructure failure")
        return result

    cli.run_domain = record
    sys.argv = [
        sys.argv[0],
        *protocol["cli_arguments"],
        "--task-ids",
        args.task,
        "--runtime-store-directory",
        str(output / "sessions"),
        "--context-capture-directory",
        str(output / "contexts"),
        "--provider-diagnostics-directory",
        str(output / "provider-diagnostics"),
        "--tau2-repository",
        protocol["tau_repository"],
        "--save-to",
        str(output / "official"),
    ]
    print(json.dumps({"starting": args.task, "output": str(output)}), flush=True)
    try:
        with (
            os.fdopen(
                os.open(output / "console.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
            ) as log,
            contextlib.redirect_stdout(log),
            contextlib.redirect_stderr(log),
        ):
            cli.main()
        verify()
        print((output / "assessment.json").read_text(), flush=True)
    except BaseException as error:
        save(output / "error.json", safe_error_diagnostic(error))
        print(json.dumps({"task": args.task, "error_type": type(error).__name__}), flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
