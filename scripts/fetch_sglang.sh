#!/usr/bin/env bash
# Check out the SGLang commit recorded in the bundled capability snapshot.
# The hash is read from the snapshot so it stays aligned with sglang_capabilities.json.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-/tmp/sglang-src}"
SNAPSHOT="${ROOT}/src/sglang_maestro/knowledge/sglang_capabilities.json"
REPO_URL="https://github.com/sgl-project/sglang"

export GIT_TERMINAL_PROMPT=0

if [[ ! -f "${SNAPSHOT}" ]]; then
  echo "错误：找不到能力快照 ${SNAPSHOT}" >&2
  exit 1
fi

COMMIT="$(
  python3 -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["sglang_commit"])' \
    "${SNAPSHOT}"
)"

if [[ -z "${COMMIT}" ]]; then
  echo "错误：快照里没有 sglang_commit" >&2
  exit 1
fi

if [[ -e "${DEST}" && ! -d "${DEST}/.git" ]]; then
  echo "错误：${DEST} 已存在但不是 git 仓库" >&2
  exit 1
fi

if [[ -d "${DEST}/.git" ]]; then
  git -C "${DEST}" fetch origin "${COMMIT}"
  git -C "${DEST}" checkout --detach "${COMMIT}"
else
  git clone "${REPO_URL}" "${DEST}"
  git -C "${DEST}" checkout --detach "${COMMIT}"
fi

echo "SGLang ${COMMIT} 已在 ${DEST}"
