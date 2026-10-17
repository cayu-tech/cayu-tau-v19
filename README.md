# Cayu V19r3 — reproduction package

This package contains the exact agent source used for the **49.742268% Pass¹** custom banking submission, the exact Cayu and Tau source commits, the saved public embedding cache, and the previously published 388 trajectories. It also contains a new portable setup wrapper. The acting code, prompts, evaluation arguments and grading logic are unchanged.

Submission: https://github.com/sierra-research/tau2-bench/pull/594

Published results: https://github.com/talgat0528/tau2-bench/releases/tag/cayu-v19r3-20261003

## Start here: verify and reproduce the submitted scores

These commands use no credentials, internet or model calls. Python 3.12/3.13 and Git are required. Run from the extracted package directory:

```sh
python3 reproduce.py verify
python3 reproduce.py score
```

Expected results:

| Metric | Result |
|---|---:|
| Tasks × trials | 97 × 4 = 388 |
| Passes / scored failures / unscored | 193 / 195 / 0 |
| Pass¹ | 49.74226804% |
| Pass² | 40.20618557% |
| Pass³ | 36.08247423% |
| Pass⁴ | 34.02061856% |

Pass¹–⁴ are reliability metrics computed across the four trials. The chronological trial pass counts are 49, 52, 47 and 45 out of 97; these are different from Pass¹–⁴. `score` reports no four-trial metrics for an incomplete or unscored matrix.

## Install a separate execution copy

The original campaign ran on Linux. The wrapper requires Linux or macOS (`fcntl` and Unix paths); Windows is not supported directly. Use Python 3.12 or 3.13, Git and `uv`. Install `uv` if needed with `python3 -m pip install uv`. Installation downloads ordinary Python packages; the exact Cayu and Tau Git snapshots are included, so their installation does not require private GitHub access.

Keep the extracted package in its final location. Choose a **new, nonexistent workspace outside the package**. For example, from the package directory:

```sh
python3 reproduce.py setup --work-dir ../v19r3-run --python python3.13
python3 reproduce.py check --work-dir ../v19r3-run
python3 reproduce.py offline-tests --work-dir ../v19r3-run
../v19r3-run/.venv/bin/python test_reproduction.py --work-dir ../v19r3-run
```

`setup` creates the virtual environment, clones the pinned Tau checkout, and copies the actor. Only `protocol.json` receives local path substitutions; all actor `.py` files and `pyproject.toml` remain byte-identical. The wrapper explicitly installs the pinned Cayu dependency from the included repository, installs all remaining declared requirements, then installs the actor with `--no-deps` so its preserved historical dependency URL is not fetched. It records each substitution and a new execution manifest in `relocation.json`, and restores the expected `.vendor/tau2` dependency link. It does not start an evaluation. `offline-tests` runs the original qualification suite and writes its receipt; it is required by the original campaign supervisor. If installation or qualification fails, retain the logs and resolve that before running the benchmark. Use a fresh workspace for a clean retry.

`constraints-verified.txt` records the dependency versions resolved for this reproduction check. **It is not the original campaign VM's complete dependency lock**, which was not preserved. Exact agent/Cayu/Tau source versions are preserved. `VALIDATION.md` records what was actually checked for this package.

## Credentials and live checks

The evaluated actor uses **GPT-5.6-sol, xhigh, through the subscription provider**. The customer uses **OpenAI Platform GPT-5.2, low**; query embeddings use **text-embedding-3-large, 3072 dimensions**. The included document vectors avoid rebuilding the stored corpus embeddings. A fresh run requires access to these exact models/providers. Running the actor on an OpenAI API key instead is available as an explicit option ([below](#optional-run-the-actor-on-an-openai-api-key)); it is a different execution setup and is never substituted silently.

Supply two files belonging to the person running the experiment, outside this package:

- A current Codex subscription `auth.json`, containing the login's `tokens` object. The frozen adapter reads `access_token`, `refresh_token`, and `account_id` and checks access-token expiry.
- A plain text file containing an OpenAI Platform API key for the simulator and query embeddings.

No credentials are included. Replace the example paths below with real paths. **The following commands make live requests and consume the runner's subscription/API allowance.** Neither installation nor offline verification makes those requests.

```sh
python3 reproduce.py live-check --work-dir ../v19r3-run \
  --auth-file /secure/path/auth.json \
  --openai-api-key-file /secure/path/openai-key.txt
```

This runs the original actor/simulator/embedding connectivity probes and eight-caller retrieval qualification. The latter produces the receipt required by the original supervisor. These checks have not been run with the recipient's credentials; model access must be verified locally.

## Optional: run the actor on an OpenAI API key

If you don't have a Codex subscription, the actor can call GPT-5.6-sol through the OpenAI API with the same API key used for the customer and embeddings. Choose this when creating the workspace; it cannot be switched on an existing one:

```sh
python3 reproduce.py setup --work-dir ../v19r3-api-run --python python3.13 --actor-provider api
python3 reproduce.py check --work-dir ../v19r3-api-run
python3 reproduce.py offline-tests --work-dir ../v19r3-api-run
../v19r3-api-run/.venv/bin/python test_reproduction.py --work-dir ../v19r3-api-run

python3 reproduce.py live-check --work-dir ../v19r3-api-run \
  --openai-api-key-file /secure/path/openai-key.txt
python3 reproduce.py run --work-dir ../v19r3-api-run \
  --openai-api-key-file /secure/path/openai-key.txt
```

`--auth-file` is not needed. `status`, `continue` and `score` work as described below.

What changes, and what doesn't:

- Agent code, prompts, model name, `xhigh` effort, limits, seeds, customer, embeddings and every other CLI argument are unchanged. The only change is `--provider subscription` → `--provider api` in the execution copy's `protocol.json`, recorded under `cli_argument_changes` in `relocation.json`. `check` and `test_reproduction.py` reject any other change.
- The actor uses Cayu's standard `OpenAIProvider` (OpenAI Responses API) instead of the Codex subscription backend. The submitted run's subscription-only recovery wrapper for interrupted streams (`RecoveringSubscriptionProvider`) is not used; Cayu's ordinary transient retries (up to 10 attempts per model call) still apply.
- The frozen supervisor's connectivity probe always uses the subscription provider, so `live-check` and `run` start the supervisor through `api_runner.py`, which replaces only that probe with an API-key probe. Workers, checkpoints, the trial-1 pause, recovery and scoring are the unchanged frozen supervisor.
- The key's organization needs access to `gpt-5.6-sol` with `xhigh` reasoning, `gpt-5.2` and `text-embedding-3-large`. All actor, customer and embedding usage is billed to that key.
- Report results from this mode as an API-provider rerun of the V19r3 agent, not as the submitted configuration.

## Run the same four-trial benchmark

After both qualifications pass:

```sh
python3 reproduce.py run --work-dir ../v19r3-run \
  --auth-file /secure/path/auth.json \
  --openai-api-key-file /secure/path/openai-key.txt
```

Run on a persistent machine/session. The command supervises all 97 base-split banking tasks, four trials and eight concurrent tasks. Each task starts a fresh conversation. It retains the original pause after trial 1. From another terminal, inspect progress and explicitly release the remaining trials:

```sh
python3 reproduce.py status --work-dir ../v19r3-run
python3 reproduce.py continue --work-dir ../v19r3-run
```

`continue` requires the first-trial boundary to have been reached and binds authorization to that execution manifest. It does not change the candidate or retry scored failures.

Results live under `../v19r3-run/actor/official/results.json`:

```sh
python3 reproduce.py score --results ../v19r3-run/actor/official/results.json
```

A new model run can differ from the original trajectories. It must be reported as a new evaluation; it does not replace the submitted results.

## Resume and recovery

The original supervisor uses upstream checkpoint resume. It preserves checkpoints in `attempt-history/` before recovery, skips **all scored results including zeros**, and retries infrastructure outcomes at their registered task/trial/seed. At most three completed infrastructure retry rounds are allowed; shared auth/quota pauses do not consume a round. Five historical infrastructure attempts were retried in the submitted campaign; no scored outcome was replaced. See `results/infrastructure-attempts.json` and `results/methodology.md`.

Shared authentication/quota failures pause supervision. Update the external credential file or allowance, then allow the existing supervisor to recheck; it can wait up to 15 minutes. If the supervisor exited, rerun the identical `run` command against the same workspace. Do not edit source, delete checkpoints, or rerun scored failures. A `STOP` file in `v19r3-run/actor/` pauses the original supervisor; removing that file permits it to continue. It also pauses when free disk falls below 3 GiB. Preserve sufficient additional disk for growing session/context logs.

## Exact configuration

| Setting | Submitted value |
|---|---|
| Agent manifest SHA-256 | `93300d01e05e9721de27e650377a270467665aecdd3792d361646d694b92714d` |
| Cayu commit | `d8e530acafa3cb7ab9596340182cffc8d9467d5d` |
| Tau commit / version | `fc0055dc4e0a316c3f83133267fbd6faaa770992` / v1.0.1 |
| Base seed | 300 |
| Trial seeds | 626729, 373753, 361454, 1567 |
| Runtime / candidate profile | native / constraints-v4 |
| Planning / advice / controller | outcomes / lean / advisory |
| Compaction trigger / warning | 180,000 / 150,000 estimated tokens |
| Actor / callback / simulation deadline | 7,200 / 7,350 / 7,500 seconds |
| Native steps / Tau steps | 192 / 200 |
| Transient model attempts | Up to 10; unknown errors are not automatically retried |
| Retrieval | Cayu hybrid retrieval; upstream `no_knowledge` because Cayu supplies retrieval |

The immutable `actor/protocol.json` and `provenance/recorded-effective-settings.json` are authoritative. The old `actor/README.md` describes an earlier development screen and remains unmodified for integrity; use this package README for reproduction.

## Contents and sharing

- `actor/`: all 357 original manifest files, listed with their SHA-256 hashes in `actor/source-manifest.json`. Some are development and diagnostic scripts from the frozen snapshot that the benchmark run does not use; they are kept byte-identical so the manifest matches the submission. Absolute paths inside frozen files refer to the original development machine; `setup` rebinds the ones the run needs.
- `dependencies/*.git/`: shallow, exact-commit Git repositories; no unrelated commit history or original remotes. Setup restores readable checkouts. The pinned Cayu source is included because its original repository requires access.
- `embedding-cache/`: preserved local public-corpus vectors and text hashes. These were not re-compared with the original VM after its preservation; integrity and source compatibility are checked locally.
- `results/`: the previously published sanitized trajectories, submission metadata, methodology and recovery summary. The trajectory JSON is gzip-compressed; its decompressed bytes match the published file.
- `reproduce.py`, `api_runner.py`, `test_reproduction.py`, `constraints-verified.txt`: new packaging/verification files, separate from evaluated actor code.
- `PACKAGE_SHA256.json`, `VALIDATION.md`: package integrity and verification record.

Original dependency license files are retained in the Git snapshots. Credentials, raw provider captures, session databases and the unpublished canonical trajectories are not included.
