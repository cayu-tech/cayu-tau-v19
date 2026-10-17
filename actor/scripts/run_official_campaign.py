"""Persistent full-domain evaluation using upstream trial, seed, grade and resume logic."""

import argparse
import asyncio
import fcntl
import gzip
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_full_benchmark import digest, verify, write_json  # noqa: E402
from trial_boundary import (  # noqa: E402
    continuation_allowed,
    first_trial_status,
    staged_trial_dispatch,
)


def expected_trials(root):
    path = root / "protocol.json"
    protocol = json.loads(path.read_text()) if path.exists() else {}
    trials = protocol.get("trials", 4)
    if type(trials) is not int or trials not in (1, 4):
        raise ValueError("Full-domain evaluation supports one or four trials")
    if protocol.get("expected_trajectories", 97 * trials) != 97 * trials:
        raise ValueError("Expected trajectory count must match all 97 tasks")
    return trials


def summary(root):
    path = root / "official/results.json"
    try:
        sims = json.loads(path.read_text()).get("simulations", []) if path.exists() else []
    except (json.JSONDecodeError, FileNotFoundError):
        # Upstream writes its checkpoint under an internal lock. A reader outside
        # that process may catch the write in progress; never crash supervision.
        return {"expected": 97 * expected_trials(root), "finished": 0, "passed": 0, "scored_failures": 0,
                "unscored": 0, "snapshot_pending": True}
    return {
        "expected": 97 * expected_trials(root),
        "finished": len(sims),
        "passed": sum((s.get("reward_info") or {}).get("reward") == 1 for s in sims),
        "scored_failures": sum((s.get("reward_info") or {}).get("reward") == 0 for s in sims),
        "unscored": sum((s.get("reward_info") or {}).get("reward") is None for s in sims),
    }


def preserve_checkpoint(root):
    source = root / "official/results.json"
    if source.exists():
        history = root / "attempt-history"
        history.mkdir(exist_ok=True)
        target = history / f"before-resume-{time.time_ns()}.json.gz"
        with source.open("rb") as src, gzip.open(target, "xb") as dst:
            shutil.copyfileobj(src, dst)


def install_runtime(root, key_file):
    from run_lean_advice_task import configure_actor_provider, configure_api_key

    from cayu_tau3_banking import cli

    protocol = json.loads((root / "protocol.json").read_text())
    configure_api_key(key_file)
    configure_actor_provider(protocol)
    return cli, protocol


def worker(root, key_file):
    verify(root)
    cli, protocol = install_runtime(root, key_file)
    from evaluate_section_embeddings import CachedEmbeddings

    original_build = cli.build_memory_corpus
    original_run = cli.run_domain

    async def cached_corpus(documents, **kwargs):
        cache_root = Path(protocol["corpus_embedding_cache"])
        cache = CachedEmbeddings(kwargs["embedding_provider"])
        manifest = json.loads((cache_root / "manifest.json").read_text())
        cache.restore(cache_root, manifest["corpus_sha256"])
        kwargs["embedding_provider"] = cache
        return await original_build(documents, **kwargs)

    def run(config):
        if (config.task_ids or config.num_tasks or config.num_trials != expected_trials(root)
                or config.seed != protocol.get("seed", config.seed)):
            raise ValueError("Campaign requires the full domain and the registered trial count")
        preserve_checkpoint(root)
        # Upstream resume skips all scored results, including failures, and retries
        # infrastructure failures with the same trial and seed. Keep the original
        # checkpoint above before upstream replaces any infrastructure records.
        with staged_trial_dispatch(root, protocol):
            result = original_run(config.model_copy(update={"auto_resume": True}))
        verify(root)
        return result

    cli.build_memory_corpus = cached_corpus
    cli.run_domain = run
    sys.argv = [sys.argv[0], *protocol["cli_arguments"],
                "--runtime-store-directory", str(root / "sessions"),
                "--context-capture-directory", str(root / "contexts"),
                "--provider-diagnostics-directory", str(root / "provider-diagnostics"),
                "--tau2-repository", protocol["tau_repository"],
                "--save-to", str(root / "official")]
    cli.main()


async def probe(root, key_file):
    from contextlib import aclosing

    from cayu import Message
    from cayu.providers.base import ModelRequest, ModelStreamEventType
    from openai import AsyncOpenAI

    cli, _ = install_runtime(root, key_file)
    provider = cli.build_provider(provider="subscription", codex_lb_base_url="",
                                  semantic_progress_timeout_seconds=45)
    try:
        async with asyncio.timeout(90):
            completed = False
            request = ModelRequest(model="gpt-5.6-sol", messages=[Message.text("user", "Reply OK.")],
                                   options={"thinking": {"enabled": True, "effort": "xhigh"}})
            async with aclosing(provider.runtime_stream(request)) as stream:
                async for event in stream:
                    if event.type is ModelStreamEventType.ERROR:
                        raise RuntimeError("Subscription connectivity failed")
                    completed |= event.type is ModelStreamEventType.COMPLETED
            if not completed:
                raise RuntimeError("Subscription completion missing")
        async with AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60, max_retries=0) as api:
            response = await api.chat.completions.create(model="gpt-5.2",
                messages=[{"role": "user", "content": "Reply OK."}], reasoning_effort="low",
                max_completion_tokens=128)
            if not response.choices:
                raise RuntimeError("Simulator connectivity failed")
            vectors = await api.embeddings.create(model="text-embedding-3-large", input="connectivity", dimensions=3072)
            if len(vectors.data[0].embedding) != 3072:
                raise RuntimeError("Embedding connectivity failed")
    finally:
        await provider.aclose()


def explicit_shared_outage(error):
    """Typed shared outages only; generic throttling is handled by bounded retries."""
    if not isinstance(error, dict):
        return False
    # HTTP-boundary captures can have an empty error object (for example a
    # non-JSON 401 response). Do not depend on a parsed provider error code.
    # Conflicting status evidence needs review, not an inferred auth outage.
    if error.get("error_status_conflict") is not True:
        statuses = [error.get("http_status_code"), error.get("status_code")]
        nested_error = error.get("error")
        if isinstance(nested_error, dict):
            statuses.extend(nested_error.get(key) for key in ("http_status_code", "status_code"))
        if any(type(value) is int and value in (401, 403) for value in statuses):
            return True
    codes = {
        "usage_limit_reached", "insufficient_quota", "billing_hard_limit_reached",
        "invalid_api_key", "token_expired", "authenticationerror", "authentication_error",
        "permissiondeniederror", "account_deactivated", "organization_deactivated",
    }
    identities = [error.get(key) for key in ("type", "error_type", "code")]
    nested = error.get("error")
    if isinstance(nested, dict):
        identities.extend(nested.get(key) for key in ("type", "error_type", "code"))
    return any(isinstance(value, str) and value.lower() in codes for value in identities)


def shared_failure(root, since):
    checkpoint = root / "official/results.json"
    try:
        sims = json.loads(checkpoint.read_text()).get("simulations", [])
    except (OSError, ValueError):
        sims = []
    for sim in sims:
        try:
            timestamp = datetime.fromisoformat(sim["timestamp"].replace("Z", "+00:00")).timestamp()
        except (KeyError, ValueError, TypeError):
            continue
        if timestamp >= since and explicit_shared_outage(sim.get("info")):
            return "provider_auth_or_quota"
    for path in (root / "provider-diagnostics").glob("*/*.error.json"):
        try:
            if path.stat().st_mtime < since:
                continue
            error = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if explicit_shared_outage(error):
            return "provider_auth_or_quota"
    return None


def stop_worker(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            # Same process group belongs exclusively to this systemd service;
            # the batch worker has threads, not detached task processes.
            process.kill()
            process.wait(timeout=30)


def supervise(root, key_file):
    verify(root)
    qualification = json.loads((root / "qualification.json").read_text())
    if (qualification.get("passed") is not True
        or qualification.get("source_manifest_sha256") != digest(root / "source-manifest.json")):
        raise ValueError("VM qualification has not passed")
    if not json.loads((root / "parallel-recall-qualification.json").read_text()).get("passed"):
        raise ValueError("Eight-caller live retrieval qualification has not passed")
    lock = (root / "campaign.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = root / "campaign-status.json"
    prior = json.loads(state_path.read_text()) if state_path.exists() else {}
    epoch = prior.get("epoch", 0)
    rounds = prior.get("completed_infrastructure_retry_rounds", 0)
    def publish(status, **extra):
        write_json(state_path, {"status": status, "epoch": epoch,
            "completed_infrastructure_retry_rounds": rounds,
            "supervisor_pid": os.getpid(), "updated_unix": time.time(), **summary(root), **extra})
    try:
        while True:
            verify(root)
            totals = summary(root)
            if totals["finished"] == totals["expected"] and not totals["unscored"]:
                publish("completed")
                return
            if rounds >= 3:
                publish("needs_infrastructure_review")
                return
            if (root / "STOP").exists() or shutil.disk_usage(root).free < 3 * 1024**3:
                publish("paused_stop_or_low_disk")
                time.sleep(30)
                continue
            publish("checking_connectivity")
            try:
                asyncio.run(probe(root, key_file))
            except Exception as error:
                publish("paused_provider_connectivity", error_type=type(error).__name__)
                time.sleep(900)
                continue
            epoch += 1
            publish("starting_worker")
            since = time.time()
            with (root / f"batch-{epoch:03d}.log").open("xb") as log:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                    "worker", "--root", str(root), "--openai-api-key-file", str(key_file)],
                    cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                reason = None
                try:
                    while process.poll() is None:
                        reason = shared_failure(root, since)
                        if (root / "STOP").exists():
                            reason = "requested_stop"
                        if shutil.disk_usage(root).free < 3 * 1024**3:
                            reason = "low_disk"
                        boundary = (
                            first_trial_status(root)
                            if json.loads((root / "protocol.json").read_text()).get("pause_after_trial") == 1
                            and not continuation_allowed(root) else None
                        )
                        if boundary is not None:
                            publish("paused_after_trial_1", worker_pid=process.pid, first_trial=boundary)
                        else:
                            publish("running", worker_pid=process.pid)
                        if reason:
                            stop_worker(process)
                            break
                        time.sleep(15)
                finally:
                    stop_worker(process)
            preserve_checkpoint(root)
            if reason:
                publish("paused", reason=reason)
                time.sleep(900 if reason.startswith("provider") else 30)
                continue
            if process.returncode:
                publish("worker_error", exit_code=process.returncode)
                raise RuntimeError("Batch process failed; service will restart from checkpoint")
            totals = summary(root)
            if totals["unscored"]:
                rounds += 1
                publish("retrying_infrastructure_only")
                time.sleep(120)
    finally:
        lock.close()


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "supervise", "probe"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--openai-api-key-file", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "worker":
        worker(root, args.openai_api_key_file)
    elif args.command == "probe":
        asyncio.run(probe(root, args.openai_api_key_file))
    else:
        supervise(root, args.openai_api_key_file)


if __name__ == "__main__":
    main()
