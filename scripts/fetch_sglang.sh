#!/usr/bin/env bash
# Check out the SGLang commit recorded in the bundled capability snapshot.
# The hash is read from the snapshot so it stays aligned with sglang_capabilities.json.
# A missing directory is a depth-1 fetch of that SHA. An existing checkout is never
# passed --depth, so a full clone stays full.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-/tmp/sglang-src}"
SNAPSHOT="${ROOT}/src/sglang_maestro/knowledge/sglang_capabilities.json"
REPO_URL="${SGLANG_REPO_URL:-https://github.com/sgl-project/sglang}"

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
  if git -C "${DEST}" cat-file -e "${COMMIT}^{commit}" 2>/dev/null; then
    git -C "${DEST}" checkout --detach "${COMMIT}"
  else
    git -C "${DEST}" fetch origin "${COMMIT}"
    git -C "${DEST}" checkout --detach "${COMMIT}"
  fi
else
  git init -q "${DEST}"
  git -C "${DEST}" remote add origin "${REPO_URL}"
  if ! git -C "${DEST}" fetch --depth 1 origin "${COMMIT}"; then
    echo "按 SHA 浅取失败，回退到完整 fetch。"
    git -C "${DEST}" fetch origin "${COMMIT}"
  fi
  git -C "${DEST}" checkout --detach FETCH_HEAD
fi

echo "SGLang ${COMMIT} 已在 ${DEST}"
