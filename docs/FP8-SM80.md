# Ampere (SM80/SM86) FP8 QSA KV backport

> **State:** source patch and CPU/static tests validated; GPU runtime, long-context
> accuracy, VRAM capacity, MTP and performance **not yet qualified**. Keep the
> existing BF16 production service until the local acceptance matrix passes.

## Scope and provenance

This is a **targeted backport** of [vLLM PR #60602](https://github.com/vllm-project/vllm/pull/60602)
(merged 2026-10-13) onto the **exact** pinned vLLM commit
`a5a30471ff2bb7f0824f2da10e358af98d304472`, **after** applying
`patches/qwen38-ple-ssd.patch`. It is not a bump of the vLLM dependency.

The new patch is `patches/qwen38-sm80-fp8-qsa-kv.patch`. It changes exactly
two files inside a *separate upstream vLLM checkout*:

- `vllm/models/qwen4_exp/nvidia/ops/qsa.py`: raw `uint8` e4m3 decode below
  SM89; bit-cast to fp16, effective ×256 host scale adjustment; SM80 tile
  tuning; warmup dtype alignment and sparse-attention dispatch.
- `vllm/models/qwen4_exp/nvidia/qsa.py`: avoid the e4m3 tensor view below SM89
  and accept raw uint8 caches.

**Intentional difference from the upstream PR:** the pinned September vLLM
has no `self.use_fused_qsa_prepare` or the October fused KV writer. Therefore
the upstream PR's additional fused-writer gate does **not** apply and was not
copied into this backport. The pinned separate `do_kv_cache_update` writer
remains in place. All eleven relevant sparse-kernel hunks and the one relevant
attention-owner hunk were matched to their exact baseline source.

Only the **main QSA KV pool** is converted to FP8. This is *not* compression
of GDN recurrent state, QSA indexer state, PLE table, model weights, or MTP
weights. An approximately 1.8× improvement to *total* vLLM KV token capacity
was observed on other Ampere deployments in upstream PR #60602; it is not an
expectation guaranteed for this 64 GiB SSD-PLE configuration.

### FP8 math and numerical caveat

On SM80/SM86 Triton cannot load the e4m3 NVFP8 pointer type used on newer
GPUs. The kernel instead loads raw `uint8`, synthesizes fp16 bits representing
`e4m3 / 256`, then folds 256 into the scalar K/V dequant scales. The
independent CPU test enumerates all 254 finite e4m3fn codes exactly.

The two e4m3 NaN codes decode to ±480 rather than NaN, consistent with the
merged upstream implementation. This requires a real-model accuracy check;
a bit-exact CPU codec test does not prove equal model outputs.

## Apply the patch to a separate checkout

Do not change the live production vLLM source or editable venv in-place.

```bash
git clone --branch feat/sm80-fp8-qsa-kv-backport --single-branch \
  https://github.com/TnzGit/qwen3.8-flash-next-cmp170hx.git ~/qwen38-fp8-lab
cd ~/qwen38-fp8-lab

export VLLM_WORKDIR="$HOME/vllm-fp8-lab"
mkdir -p "$VLLM_WORKDIR"
git clone https://github.com/vllm-project/vllm.git "$VLLM_WORKDIR/src"
git -C "$VLLM_WORKDIR/src" checkout a5a30471ff2bb7f0824f2da10e358af98d304472

bash scripts/apply-patch.sh
bash scripts/apply-fp8-kv.sh
bash scripts/apply-fp8-kv.sh  # safe / idempotent
git -C "$VLLM_WORKDIR/src" diff --check
VLLM_SOURCE="$VLLM_WORKDIR/src" python3 -m unittest discover -s tests \
  -p 'test_fp8_sm80_backport.py' -v
```

On a **fresh isolated workspace** only, `bash scripts/install-runtime.sh`
installs the pinned runtime and both patches. The repository's original runtime
installer was not originally a bit-for-bit reconstruction of the production
environment; first verify the wheel/toolkit and library paths. Do not run the
generic model downloader when using the existing *uncensored* checkpoint.

## Switch

The launcher now forwards `QWEN_KV_CACHE_DTYPE` to vLLM's
`--kv-cache-dtype`; it defaults to `auto` (existing BF16 behavior).
Examples for the existing CMP170HX service profile, **on the isolated copy**:

```bash
export QWEN_MODEL_DIR="$HOME/models/Qwen3.8-Flash-Next-Uncensored-AutoRound-3bpw-MTP"
export VLLM_WORKDIR="$HOME/vllm-fp8-lab"
QWEN_PRINT_COMMAND=1 QWEN_KV_CACHE_DTYPE=fp8 \
  bash scripts/serve-cmp170hx-profile.sh
```

Use `QWEN_HOST=127.0.0.1 QWEN_PORT=8003` for the test server. The production
profile defaults to context 243200, `max-num-seqs=1`, memory utilization
0.90, MTP 3, and graph captures `[4]`. Do **not** change all of these
simultaneously. To attempt concurrent long contexts, later increase
`QWEN_SEQS=2` and test capacity; two 240K contexts require about 480K tokens'
worth of concurrent state plus overhead and are **not** automatically guaranteed
to fit with FP8.

## Verification performed without GPU

GitHub Actions workflow:
[`verify-fp8-backport.yml`](../.github/workflows/verify-fp8-backport.yml)

- Checks out the exact pinned upstream vLLM SHA.
- Applies existing SSD-PLE patch, then this FP8 patch; verifies idempotence
  and `git diff --check`.
- Checks shell syntax and Python syntax.
- Runs exhaustive finite e4m3 byte-conversion and backport structure tests.
- Verifies that the opt-in flag reaches the launcher in dry-run mode.

This **does not** compile Triton, allocate FP8 KV, perform a CUDA Graph
capture, test the PLE asynchronous host break, or run real Qwen inference.

## Runtime test and rollback

The local agent should follow [`handover.md`](../handover.md) for the BF16
control, FP8 smoke/accuracy matrix, MTP 3, 240K context, dual cold
sessions, and latency/VRAM measurement. Use
`benchmarks/bench_fp8_kv_capacity.py` for independent cold prompt throughput
(no quality claims). Keep host/port/network restricted during tests.

Rollback is simply to **stop the isolated FP8 service**, restore the old
systemd production user service unchanged, and start it. If backport changes
must be removed from the *isolated* vLLM source, with the FP8 service stopped:

```bash
cd ~/vllm-fp8-lab/src
git apply --reverse ~/qwen38-fp8-lab/patches/qwen38-sm80-fp8-qsa-kv.patch
```

Do not reverse the original SSD-PLE patch in the live production worktree.
