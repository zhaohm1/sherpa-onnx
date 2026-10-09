#!/usr/bin/env bash
set -euo pipefail

archive="${1:-/mnt/c/Users/ZhaoHM/Downloads/2.4.2a15.tgz}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${repo_root}/.venv/bin/python"

if [[ ! -f "${archive}" ]]; then
  echo "RKNN-Toolkit2 archive not found: ${archive}" >&2
  exit 2
fi
if [[ ! -x "${python_bin}" ]]; then
  echo "Create the project environment first: uv sync --python 3.10" >&2
  exit 2
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT
tar -xzf "${archive}" -C "${tmpdir}" \
  2.4.2a15/packages/rknn_toolkit2-2.4.2a15-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
wheel="${tmpdir}/2.4.2a15/packages/rknn_toolkit2-2.4.2a15-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
uv pip install --python "${python_bin}" --no-deps "${wheel}"
"${python_bin}" -c 'import rknn; from importlib.metadata import version; print("rknn-toolkit2", version("rknn-toolkit2")); print("import rknn: OK")'
