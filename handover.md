# Local agent handover — SM80 FP8 QSA KV on CMP 170HX

**Date:** 2026-11-01 (Asia/Singapore)  
**Repository:** https://github.com/TnzGit/qwen3.8-flash-next-cmp170hx  
**Branch:** `feat/sm80-fp8-qsa-kv-backport` (check the current tip with `git rev-parse HEAD`)  
**Upstream change:** https://github.com/vllm-project/vllm/pull/60602 (merged 2026-10-13)  
**Upstream pinned vLLM:** `a5a30471ff2bb7f0824f2da10e358af98d304472`  
**State at handoff:** GitHub source changes and non-GPU tests prepared; **NO local-GPU validation**. The production BF16 service was **not modified**.

## Mission and non-negotiable safety rules

Qualify FP8 **main QSA KV** for the existing SSD-PLE vLLM variant on a **single
unlocked 64 GiB CMP 170HX, SM80**, while preserving the uncensored AutoRound
3bpw+MTP checkpoint and current 250K-class context behavior. Measure whether
a second **independent, cold** long-context request actually fits; do not claim
a guaranteed 2×250K result from the upstream 1.83× capacity report.

1. Preserve the current production vLLM source, virtualenv, GPU settings and
   systemd service. Use a **separate** `~/vllm-fp8-lab` checkout/venv and the
   experiment branch in `~/qwen38-fp8-lab`. Do not run an installer against
   `~/vllm`; the deployed package is editable and sharing a source tree risks
   changing production unexpectedly.
2. **Only run GPU inference after intentionally stopping the production
   service**; the single 64GiB GPU cannot accommodate both complete instances
   simultaneously. Take baseline logs, record baseline configuration and
   restore production promptly on failure. Never enable the FP8 change in the
   persistent user service until all quality/performance checks pass.
3. The test endpoint must bind to `127.0.0.1` (port `8003`), not the
   production `0.0.0.0:8002`, and use no public exposure or reverse proxy.
4. No model download is needed for the existing uncensored checkpoint. Verify
   the local model revision before reuse; do not replace it with the
   default *censored* checkpoint accidentally.
5. Preserve quality as a gate: numerical codec parity and successful GPU
   startup alone are insufficient. Test long-context retrieval, tool calls,
   BF16-vs-FP8 outputs, PLE SSD I/O and MTP in realistic workloads.

## What software work is already present

- `patches/qwen38-sm80-fp8-qsa-kv.patch`: backports the relevant
  **11 ops/qsa.py hunks + 1 qsa.py hunk** from vLLM #60602 to this pinned
  vLLM SHA, *without changing* the existing 55KB SSD PLE patch. It supports
  raw uint8 E4M3 on SM80/SM86, exact finite-code bit decode + host scales,
  correct warmup cache dtype, and the SM80 configuration table adjustment.
- `scripts/apply-fp8-kv.sh`: strict base-sha check, requires SSD PLE first,
  `git apply --check`, repeat-safe, no `--3way`.
- `scripts/install-runtime.sh`: applies SSD PLE followed by FP8 backport on
  a new install. Do not run in the production worktree.
- `scripts/serve.sh` and `scripts/serve-cmp170hx-profile.sh`: expose
  `QWEN_KV_CACHE_DTYPE`, **default `auto`** (old BF16 behavior); explicit
  `fp8` opts in.
- `tests/test_fp8_sm80_backport.py`: exhaustive CPU reference for all 254
  finite E4M3 bytes, both NaN-code behavior, source AST/integration checks and
  launch-config opt-in regression.
- `.github/workflows/verify-fp8-backport.yml`: clones exact pinned vLLM,
  applies both patches, reruns the FP8 patch idempotently, checks syntax and
  CPU math, and checks command wiring.
- `benchmarks/bench_fp8_kv_capacity.py`: reproducible 8K–240K cold
  **independent token-ID** prompts, 1/2 request concurrency, SSE latency
  metrics, aggregate throughput and error recording. Random IDs are a
  capacity/compute test, **NOT** a model-quality test.
- `docs/FP8-SM80.md`: technical explanation and rollback guidance.

**Important upstream-port nuance:** the September pinned QSA owner does
*not* contain the October upstream fused-QSA-prepare path. The additional
fused-writer gating hunk of #60602 was therefore intentionally not backported;
the existing separate `do_kv_cache_update` writer is retained. Do not
blindly import the rest of the newer QSA file, which would drag in unrelated
refactors and may conflict with the existing SSD PLE/CUDA-graph patch.

**Reference evidence from upstream** (not this GPU): the merged #60602
reports ~1.83× KV token capacity (8×A100) and ~1.87× (4×RTX3090), with
GPU correctness/accuracy experiments. It only compresses QSA main KV, not
GDN state, side indexer caches, model weights or the SSD PLE table.

## Target production profile — DO NOT overwrite

File `docs/CMP170HX-SERVICE.md` describes the currently recorded service:

- Name: `qwen38-uncensored-autoround-8002.service` (systemd user service,
  `Restart=on-failure`, linger). Verify this name on the live host first.
- Uncensored model: `klee100/Qwen3.8-Flash-Next-Uncensored-AutoRound-3bpw-MTP`;
  revision `a141dff69a3f53367bfee9f36f33dfe5b2964041`.
- vLLM: pinned `a5a30471...` with `qwen38-ple-ssd.patch`; reported
  version `0.29.1rc1.dev402+ga5a30471f.ple1`.
- Context **243200**, `max-num-seqs=1`, prefill batch tokens 1024,
  `gpu-memory-utilization=0.90`, **MTP=3**, `FULL_AND_PIECEWISE` capture
  `[4]`, prefix caching, `mamba-cache-mode=align`.
- Default host/port `0.0.0.0:8002`, power limit 180W; PLE on SSD.
- The profile checks **35 safetensors shards**, sum **153012759768 bytes**,
  including two MTP shards. Keep existing files; do not rely on sum alone as a
  weight-integrity guarantee.

### 0 — Capture and preserve the live baseline

Before altering or stopping anything, record:

```bash
date -Is
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,memory.used,power.limit,driver_version --format=csv
systemctl --user status qwen38-uncensored-autoround-8002.service
systemctl --user cat qwen38-uncensored-autoround-8002.service
git -C "$HOME/vllm/src" rev-parse HEAD
git -C "$HOME/vllm/src" status --short
curl -sf http://127.0.0.1:8002/health
curl -s http://127.0.0.1:8002/metrics | grep -E 'kv_cache|num_requests'
```

Save the outputs and current logs **outside** the repo (e.g. `~/qwen38-fp8-evidence/`);
never commit user prompts, API responses containing private material,
GPU machine secrets, or raw service config credentials. If paths/service name
differ, adapt to the live host rather than guessing. Note any uncommitted
production vLLM changes not accounted for by the SSD PLE patch.

## 1 — Prepare isolated lab (while production stays running)

```bash
git clone --branch feat/sm80-fp8-qsa-kv-backport --single-branch \
  https://github.com/TnzGit/qwen3.8-flash-next-cmp170hx.git "$HOME/qwen38-fp8-lab"
cd "$HOME/qwen38-fp8-lab"
git rev-parse HEAD

export VLLM_WORKDIR="$HOME/vllm-fp8-lab"
export QWEN_MODEL_DIR="$HOME/models/Qwen3.8-Flash-Next-Uncensored-AutoRound-3bpw-MTP"
mkdir -p "$VLLM_WORKDIR"
git clone https://github.com/vllm-project/vllm.git "$VLLM_WORKDIR/src"
git -C "$VLLM_WORKDIR/src" checkout a5a30471ff2bb7f0824f2da10e358af98d304472
bash scripts/apply-patch.sh
bash scripts/apply-fp8-kv.sh
bash scripts/apply-fp8-kv.sh   # must report already applied
git -C "$VLLM_WORKDIR/src" diff --check
VLLM_SOURCE="$VLLM_WORKDIR/src" python3 -m unittest discover -s tests \
  -p 'test_fp8_sm80_backport.py' -v
```

Check installed Python 3.12, CUDA toolkit, CUDA 13 NVRTC, driver and
SSD-PLE helper build requirements against `docs/CMP170HX-SERVICE.md`.
On this **new** workdir only, install the pinned wheel/editable source and
build the SSD native AIO helper:

```bash
cd "$HOME/qwen38-fp8-lab"
export VLLM_WORKDIR="$HOME/vllm-fp8-lab"
bash scripts/install-runtime.sh
QWEN_PRINT_COMMAND=1 QWEN_KV_CACHE_DTYPE=fp8 \
  QWEN_HOST=127.0.0.1 QWEN_PORT=8003 \
  QWEN_MODEL_DIR="$HOME/models/Qwen3.8-Flash-Next-Uncensored-AutoRound-3bpw-MTP" \
  bash scripts/serve-cmp170hx-profile.sh
```

The generic installer was not originally an exact recreation of the running
host's Python environment. If wheels/dependencies differ or fail, **stop and
compare** existing production environment requirements; do not modify or
reinstall that production venv to work around an isolated-lab issue.

## 2 — GPU admission and BF16 control (exclusive access)

Do not start until baseline evidence is captured and lab preflight passes.

```bash
systemctl --user stop qwen38-uncensored-autoround-8002.service
systemctl --user is-active qwen38-uncensored-autoround-8002.service
nvidia-smi

cd "$HOME/qwen38-fp8-lab"
export VLLM_WORKDIR="$HOME/vllm-fp8-lab"
export QWEN_MODEL_DIR="$HOME/models/Qwen3.8-Flash-Next-Uncensored-AutoRound-3bpw-MTP"
export QWEN_HOST=127.0.0.1 QWEN_PORT=8003
export QWEN_KV_CACHE_DTYPE=auto
bash scripts/serve-cmp170hx-profile.sh
```

Run foreground in a dedicated terminal / SSH screen session. From a second
terminal, poll `http://127.0.0.1:8003/health` and then run
`benchmarks/verify_server.py --url http://127.0.0.1:8003 --long --tools
--output ...`. Compare accuracy and speed to the old baseline, watching the
SSD PLE logs and CUDA Graph host-break behavior.

**Do not use systemd `Restart=on-failure` for the experiment.** Stop the
foreground lab service cleanly before changing its dtype.

## 3 — Enable FP8 and run smoke/accuracy tests

Re-run the *same isolated code*, same profile, but:

```bash
export QWEN_KV_CACHE_DTYPE=fp8
bash scripts/serve-cmp170hx-profile.sh
```

Inspect startup exceptions and log the exact cache dtype and advertised
GPU KV capacity. Confirm the FP8 Triton sparse attention kernel compiles on
SM80, the separate cache writer accepts uint8, warmup works at the real
`FULL_AND_PIECEWISE` graph sizes, SSD PLE host breaks remain correct and
MTP=3 produces valid output. Test both short decode and multi-turn tool use.

Quality tests: zero-temperature BF16 vs FP8 paired prompts, long-context
passkey/needle retrieval at 32K, 128K, 220K and >=240K; at least three
needle locations (front, middle, end); both English and Chinese cases;
long multi-turn continuation, repeated prefix reuse, tool calls, and
cancellation/resume. Record exact prompt length from usage. Do not
equate *request completed* with *answer correct*. The existing repo records
a separate arithmetic inconsistency in both tested graph modes: compare to
the same BF16 baseline rather than attributing all errors to FP8.

## 4 — Memory capacity and actual two-session concurrency

Start with the production limits `QWEN_CONTEXT=243200`,
`QWEN_SEQS=1`, `QWEN_MTP=3`, memory target 0.90 and graphs `[4]`.
Collect vLLM startup lines for GPU cache blocks, total KV-token capacity,
maximum concurrency estimate, GDN state sizing and any allocation warnings.

**Only after 1-session parity passes**, restart with
`QWEN_SEQS=2` while holding all else fixed. Ramp cold/independent prompts:

| Prompt tokens each | Concurrent | BF16 control | FP8 trial |
|---:|---:|---|---|
| 8,192 | 1, 2 | smoke | smoke |
| 32,768 | 1, 2 | benchmark | benchmark |
| 131,072 | 1, 2 | benchmark | benchmark |
| 220,000 | 1, 2 | capacity control | capacity trial |
| 240,000 | 1, 2 | capacity control | **target** |

Use the new deterministic tool, keeping other flags constant:

```bash
"$VLLM_WORKDIR/.venv/bin/python" benchmarks/bench_fp8_kv_capacity.py \
  --url http://127.0.0.1:8003 --label fp8-mtp3-graph4 \
  --output "$HOME/qwen38-fp8-evidence/fp8-capacity.json" \
  --lengths 8192 32768 131072 220000 240000 \
  --concurrency 1 2 --rounds 2 --output-tokens 256
```

When producing the BF16 control, label and save to a separate file.
The script generates random token IDs with **distinct deterministic per-request
prefixes** to avoid artificial prefix-cache sharing; this does not test model
quality. Record which requests are rejected vs waiting vs preempted, plus
`vllm:kv_cache_usage_perc`, running/waiting requests, PLE AIO latency,
TTFT, TPOT, aggregate output tok/s, VRAM peak and errors. Additional 1K
output-token runs are needed if 256-token generation succeeds.

If dual 240K does not fit, do not declare failure of the backport. Record
the exact FP8 token capacity and derive whether remaining pressure is
GDN states, the QSA pool, graph capture or MTP reserves. Optionally test
`QWEN_MTP=0` and a minimal capture-mode control **one variable at a time**.
Do not increase `QWEN_GPU_MEMORY` blindly until OOM; log the
baseline free memory and reserve.

## 5 — Acceptance criteria and evidence

Do not adopt into production unless all are true:

- GPU startup, FP8 prefill/decode and real CUDA graphs work with SSD PLE,
  no fatal Triton compiler path / illegal memory access / data corruption.
- Short and long BF16/FP8 quality checks meet an agreed tolerance; log any
  changed passkey/tool-call answer with an exact reproducer.
- `max-num-seqs=2` (if target fits) demonstrably executes two independent
  requests simultaneously without OOM/preemption or silent re-prefill;
  otherwise report the **actual** concurrency ceiling, not theoretical ×1.8.
- Throughput, per-request decode, TTFT and p95 latency are captured for
  identical parameters; note MTP acceptance and warm/cold cache differences.
- Cancel-and-restart and multiple rounds do not leak GPU buffers or leave
  dead SSD-AIO tasks. Check `nvidia-smi`, API errors and the kernel logs.
- A complete rollback to the original BF16 8002 user service has been tested.

Deliver evidence as an issue/PR comment or commit a **sanitized**
`docs/FP8-LOCAL-RESULTS.md` with GPU/driver/CUDA, branch SHA,
model revision, flags, log excerpts, acceptance checks, achieved context
and per-request scores. Leave production configuration unchanged unless
separately approved.

## 6 — Rollback / resume production

Stop foreground `8003` vLLM. Ensure its process has exited and GPU memory
has been released. Then run:

```bash
systemctl --user start qwen38-uncensored-autoround-8002.service
systemctl --user status qwen38-uncensored-autoround-8002.service
curl -sf http://127.0.0.1:8002/health
```

In the **isolated lab only**, the FP8 source patch can be undone with `git -C "$HOME/vllm-fp8-lab/src" apply --reverse "$HOME/qwen38-fp8-lab/patches/qwen38-sm80-fp8-qsa-kv.patch"`.
No SSD-PLE rollback is required to recover the unchanged production service.

## Open issues / boundaries of this handoff

- No local SM80 CUDA/Triton execution, 243200-token run, 2×240K proof, AIO
  stress, model quality check or benchmark has yet been performed.
- No claim that FP8 automatically creates two full-length request slots.
- `QWEN_SEQS=1` is still the current production default, so changing
  only the KV dtype cannot produce request concurrency.
- The source patch intentionally assumes *exactly* the pinned vLLM SHA. It is
  not forward-compatible with arbitrary upstream QSA refactors.
- The existing service doc reports that its installed environment was
  assembled partly by copying packages; test the clean isolated wheel setup
  and treat environment mismatches as blockers to qualification, not reasons
  to modify production.
- CPU checks are not a substitute for correctness of FP8 cache writers,
  dtype conversions, MTP speculative branches, GDN/KV scheduling or
  actual SM80 sparse-attention kernels.
