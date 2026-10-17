#!/bin/sh
set -eu

readonly TAU2_COMMIT="fc0055dc4e0a316c3f83133267fbd6faaa770992"
readonly TAU2_REPOSITORY="https://github.com/sierra-research/tau2-bench.git"
readonly SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
readonly BENCHMARK_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly VENV_DIR="${BENCHMARK_DIR}/.venv"
readonly TAU2_DIR="${BENCHMARK_DIR}/.vendor/tau2"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  python3 -m venv "${VENV_DIR}"
fi

if [ ! -d "${TAU2_DIR}/.git" ]; then
  if [ -e "${TAU2_DIR}" ]; then
    echo "Refusing to replace non-git path: ${TAU2_DIR}" >&2
    exit 1
  fi
  mkdir -p "$(dirname -- "${TAU2_DIR}")"
  git clone --filter=blob:none --no-checkout "${TAU2_REPOSITORY}" "${TAU2_DIR}"
  git -C "${TAU2_DIR}" sparse-checkout init --cone
  git -C "${TAU2_DIR}" sparse-checkout set \
    src/tau2 \
    data/tau2/domains/banking_knowledge \
    data/tau2/user_simulator
  git -C "${TAU2_DIR}" checkout --detach "${TAU2_COMMIT}"
fi

actual_commit="$(git -C "${TAU2_DIR}" rev-parse HEAD)"
if [ "${actual_commit}" != "${TAU2_COMMIT}" ]; then
  echo "tau2 source mismatch: expected ${TAU2_COMMIT}, got ${actual_commit}" >&2
  exit 1
fi
if [ -n "$(git -C "${TAU2_DIR}" status --porcelain --untracked-files=no)" ]; then
  echo "tau2 source checkout has tracked modifications: ${TAU2_DIR}" >&2
  exit 1
fi

"${VENV_DIR}/bin/python" -m pip install -e "${TAU2_DIR}[knowledge]"
"${VENV_DIR}/bin/python" -m pip install -e "${BENCHMARK_DIR}[dev]"

"${VENV_DIR}/bin/python" -m cayu_tau3_banking.provenance \
  --tau2-repository "${TAU2_DIR}"
