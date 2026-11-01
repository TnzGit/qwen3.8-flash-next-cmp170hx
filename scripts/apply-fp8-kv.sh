#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Backport vLLM #60602 to the pinned Qwen3.8 SSD-PLE deployment.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

ple_patch="$REPO_DIR/patches/qwen38-ple-ssd.patch"
fp8_patch="$REPO_DIR/patches/qwen38-sm80-fp8-qsa-kv.patch"
cd "$VLLM_SOURCE"

if [[ $(git rev-parse HEAD) != "$VLLM_REVISION" ]]; then
    echo "Expected vLLM commit $VLLM_REVISION (do not apply to a different version)." >&2
    exit 1
fi
if ! git apply --reverse --check "$ple_patch" 2>/dev/null; then
    echo "SSD PLE patch is required first: run scripts/apply-patch.sh." >&2
    exit 1
fi
if git apply --reverse --check "$fp8_patch" 2>/dev/null; then
    echo 'Ampere FP8 QSA KV backport is already applied.'
    exit 0
fi

# Fail closed on mismatched sources or partial changes; do not use --3way.
git apply --check "$fp8_patch"
git apply "$fp8_patch"
git diff --check
echo 'Applied Ampere FP8 QSA KV backport. Opt in with QWEN_KV_CACHE_DTYPE=fp8.'
