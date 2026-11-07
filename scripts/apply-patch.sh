#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
cd "$VLLM_SOURCE"
for patch in "$REPO_DIR/patches/qwen38-ple-ssd.patch" \
             "$REPO_DIR/patches/qwen3-parser-tool-call-recovery.patch"; do
    if git apply --reverse --check "$patch" 2>/dev/null; then
        echo "Already applied: $(basename "$patch")"
        continue
    fi
    if [[ $(git rev-parse HEAD) != "$VLLM_REVISION" ]]; then
        echo "Expected vLLM revision $VLLM_REVISION; check out that revision first." >&2
        exit 1
    fi
    git apply --check "$patch"
    git apply "$patch"
done
git diff --check
