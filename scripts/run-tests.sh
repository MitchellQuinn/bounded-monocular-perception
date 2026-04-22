#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Expected virtualenv Python at ${PYTHON_BIN}" >&2
  exit 1
fi

run_pytest() {
  local workdir="$1"
  local pythonpath="$2"

  echo
  echo "==> ${workdir}"

  if [[ -n "${pythonpath}" ]]; then
    (
      cd "${REPO_ROOT}/${workdir}"
      PYTHONPATH="${pythonpath}" "${PYTHON_BIN}" -m pytest tests
    )
  else
    (
      cd "${REPO_ROOT}/${workdir}"
      "${PYTHON_BIN}" -m pytest tests
    )
  fi
}

run_pytest "02_synthetic-data-processing-v3.0" "."
run_pytest "03_rb-training-v2.0" "."
run_pytest "04_ROI-FCN/01_preprocessing" "src:tests:../../02_synthetic-data-processing-v3.0"
run_pytest "04_ROI-FCN/02_training" "src:tests:../../02_synthetic-data-processing-v3.0"
run_pytest "05_inference-v0.1" ""
run_pytest "05_inference-v0.2" ""
