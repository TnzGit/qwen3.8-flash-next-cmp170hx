# CMP 170HX service profile

This profile records the running single-GPU service inspected on 2026-11-01. It uses the uncensored checkpoint, not the default checkpoint in the general guide. No model weights are included here.

* Model: `klee100/Qwen3.8-Flash-Next-Uncensored-AutoRound-3bpw-MTP`, revision `a141dff69a3f53367bfee9f36f33dfe5b2964041`.
* vLLM: upstream `a5a30471ff2bb7f0824f2da10e358af98d304472`, with this repository's PLE patch (SHA-256 `09c4e13fbd2b77169a7f8cb28d36242cc6e3916fe9c5a3c25d8e90101a38ccb0`). The installed version reported `0.29.1rc1.dev402+ga5a30471f.ple1`. The checked-out source passed reverse-patch validation; that alone does not prove no other changes exist.
* Hardware/software: one 64 GiB CMP 170HX, Linux x86-64, Python 3.12, driver 610.43.02, torch 2.13.0, transformers 5.17.0. The server uses SSD PLE offload, so the entire checkpoint must remain available on local storage while serving.
* HTTP on `0.0.0.0:8002`, no API key; restrict network access appropriately. Model ID `Qwen3.8-Flash-Next`; context limit 243200, one sequence, batch tokens 1024, GPU utilization 0.90, MTP 3, FULL_AND_PIECEWISE graph capture `[4]`, prefix caching, prompt token details. Power limit 180 W.

## Provision

Follow the source checkout and runtime installation steps in [GUIDE.md](GUIDE.md), but **do not run its model download step**: it selects a different model and revision. The original installation copied packages from an existing environment; the provided installer and package snapshot are not a verified bit-for-bit reconstruction of that environment. Check wheel availability and validate on the target machine.

```bash
export VLLM_WORKDIR="$HOME/vllm"
export QWEN_MODEL_DIR="$HOME/models/Qwen3.8-Flash-Next-Uncensored-AutoRound-3bpw-MTP"
mkdir -p "$QWEN_MODEL_DIR"
"$VLLM_WORKDIR/.venv/bin/hf" download klee100/Qwen3.8-Flash-Next-Uncensored-AutoRound-3bpw-MTP --revision a141dff69a3f53367bfee9f36f33dfe5b2964041 --local-dir "$QWEN_MODEL_DIR"
QWEN_PRINT_COMMAND=1 bash scripts/serve-cmp170hx-profile.sh
bash scripts/serve-cmp170hx-profile.sh
```

The profile checks 35 safetensors files and total shard size 153012759768 bytes; this is not a cryptographic integrity check. CUDA 13 NVRTC libraries and headers are exported from the Python 3.12 environment. Verify `/health` and `/v1/models` on port 8002 after model and graph initialization; a short completion should also be checked. Long-context and memory behavior require qualification on the target hardware.

## Autostart

The running machine uses a systemd **user** service, `Restart=on-failure`, and user linger. Adapt this template with absolute paths to the checked-out repository, vLLM workdir and model directory; systemd does not expand shell variables in `ExecStart`.

```ini
[Unit]
Description=Qwen3.8 Flash Next CMP170HX on port 8002
After=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/qwen3.8-flash-next-cmp170hx
Environment=VLLM_WORKDIR=%h/vllm
Environment=QWEN_MODEL_DIR=%h/models/Qwen3.8-Flash-Next-Uncensored-AutoRound-3bpw-MTP
ExecStartPre=/usr/bin/sudo -n /usr/bin/nvidia-smi -i 0 -pl 180
ExecStart=/usr/bin/bash %h/qwen3.8-flash-next-cmp170hx/scripts/serve-cmp170hx-profile.sh
Restart=on-failure
RestartSec=20
TimeoutStopSec=120
KillMode=control-group

[Install]
WantedBy=default.target
```

Install as `~/.config/systemd/user/qwen38-uncensored-autoround-8002.service`, run `systemctl --user daemon-reload` and `systemctl --user enable --now qwen38-uncensored-autoround-8002.service`, and enable lingering with `sudo loginctl enable-linger "$USER"`. The power-limit step requires a narrowly scoped passwordless sudo rule for the exact `nvidia-smi` command, or a separate privileged boot-time power-limit service; without it the unit will not start. Verify with `systemctl --user status`, `loginctl show-user "$USER" -p Linger`, `nvidia-smi -q -d POWER`, and the HTTP endpoints. This template is a portable reconstruction, not a boot-tested copy of the original host configuration.

## Experimental SM80 FP8 QSA KV (separate branch; not production-qualified)

The `feat/sm80-fp8-qsa-kv-backport` branch adds a targeted vLLM #60602
backport and `QWEN_KV_CACHE_DTYPE` to the launcher; default is `auto`, preserving
this profile's BF16 behavior. Set `QWEN_KV_CACHE_DTYPE=fp8` only in a separate
checkout and Python environment after applying both patches. Do not alter this
systemd service or its editable vLLM source in-place. The existing profile still
limits `QWEN_SEQS=1`: to test true concurrency, explicitly set `QWEN_SEQS=2`
in the isolated service and use the [handover test matrix](../handover.md).
The upstream 1.83× KV capacity result does not establish 2×243200 concurrency
on this 64 GiB card. See [FP8-SM80.md](FP8-SM80.md).
