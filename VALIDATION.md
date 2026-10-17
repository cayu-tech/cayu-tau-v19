# Validation record — October 17, 2026

The final package was checked on macOS arm64, CPython 3.13.1, in a fresh temporary virtual environment whose path contains spaces. No actor, customer or embedding API calls were made; no new benchmark was started.

- All 357 archived actor-manifest files match the original source hashes.
- Both included Git snapshots pass full object verification at the recorded commits. They contain only the pinned snapshot, with shallow history boundaries and no original remotes.
- Fresh installation succeeds using the included exact Cayu and Tau sources and the supplied newly resolved dependency constraints. Installed-package consistency check passes for all 87 packages.
- The original qualification suite passes **1,950 tests, zero failures, zero errors, zero skips**. Its unmodified tests cover the original supervisor/checkpoint and trial-boundary behavior as well as actor logic.
- The package scorer matches the pinned upstream Tau scorer on all four published Pass¹–⁴ metrics. Duplicate task/trial slots are rejected; incomplete or unscored matrices do not receive full four-trial scores.
- All 3,672 preserved cached texts restore, and the actual corpus builds with zero embedding API calls and the expected corpus hash.
- Relocation preserves all acting code, prompts, original pyproject bytes and CLI arguments. Only protocol.json's three external paths are changed in the new execution copy. The original packaged source stays unchanged.
- A known-credential scan covers the package, decoded published trajectories and expanded exact dependency source trees; no known credential values were found. Only the published sanitized trajectories are included, not the private canonical/provider captures.

The installer keeps the archived pyproject.toml byte-identical, including its historical dependency URL, and installs the exact included Cayu dependency explicitly before installing the actor with --no-deps. No test was removed, relaxed or skipped.

These checks establish source integrity, installation and offline behavior. They are not a new Linux live campaign, a guarantee that the recipient has model access, a replay of every original provider request, or a promise of identical new model outputs. The original VM's full transitive dependency lock was not preserved; constraints-verified.txt describes this new validated installation. The document-vector cache is the preserved local configured copy, not newly compared to the historical VM. Live actor/customer/query-embedding checks must be run with the recipient's own credentials using the documented command.

The optional API-key actor mode (`--actor-provider api`, `api_runner.py`) was checked separately on macOS arm64 with CPython 3.12, in a fresh workspace set up with `--actor-provider api`: setup and `check` pass; the offline qualification suite passes 1,950 tests with zero failures, errors or skips (run with two test shards instead of eight); `test_reproduction.py` passes; and `live-check` passes with an OpenAI API key, covering the actor (`gpt-5.6-sol`, `xhigh`, through the API), the customer model, query embeddings and the eight-caller retrieval check. No benchmark task was run in this mode.

The package checksum manifest is PACKAGE_SHA256.json.
