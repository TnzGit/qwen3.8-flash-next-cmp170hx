#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Apply/reverse an additive experiment on the pinned, already PLE-patched vLLM.

Default: validate only. Never checks out commits, changes production launchers,
installs packages, or overwrites an unrelated edit. Uses git apply atomically.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
from pathlib import Path
import subprocess

PIN = "a5a30471ff2bb7f0824f2da10e358af98d304472"
SOURCE_BLOB = "d94702612d9ba01f033e751592f93881dc991f6d"
HOOK_PATH = "vllm/v1/worker/gpu/spec_decode/mtp/speculator.py"
MODULE_PATH = "vllm/models/qwen4_exp/nvidia/mtp_draft_vocab.py"
ANCHOR = "        draft_model = load_eagle_model(target_model, self.vllm_config)\n"
HOOK = '''        if self.vllm_config.additional_config.get("qwen4_mtp_draft_vocab"):
            from vllm.models.qwen4_exp.nvidia.mtp_draft_vocab import install

            install(draft_model, self.vllm_config)
'''


def git(root: Path, *args: str, data: str | None = None) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], input=data,
                            text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def blob_sha(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode()
    return hashlib.sha1(header + raw).hexdigest()


def make_diff(before: str, after: str, path: str, new: bool = False) -> str:
    header = f"diff --git a/{path} b/{path}\n"
    if new:
        header += "new file mode 100644\n"
    return header + "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile="/dev/null" if new else f"a/{path}", tofile=f"b/{path}"))


def prepare(root: Path, runtime: str) -> tuple[str, bool]:
    """Validate just the touched files; the existing PLE patch may be uncommitted."""
    source_path, module_path = root / HOOK_PATH, root / MODULE_PATH
    for path in (source_path, module_path):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"refusing a symlink/out-of-checkout path: {path}")
    current_raw = source_path.read_bytes()
    current = current_raw.decode("utf-8")
    if current.count(ANCHOR) != 1:
        raise ValueError("MTP loading anchor is missing or ambiguous")
    installed = ANCHOR + HOOK in current
    before = current.replace(ANCHOR + HOOK, ANCHOR, 1) if installed else current
    if blob_sha(before.encode("utf-8")) != SOURCE_BLOB:
        raise ValueError("MTP speculator differs from the audited pinned source; refusing edits")
    if installed:
        if not module_path.exists() or module_path.read_bytes() != runtime.encode("utf-8"):
            raise ValueError("installed module is missing/modified; refusing overwrite or removal")
    elif module_path.exists():
        raise ValueError("experiment module already exists without the expected hook")
    after = before.replace(ANCHOR, ANCHOR + HOOK, 1)
    patch = make_diff(before, after, HOOK_PATH)
    patch += make_diff("", runtime, MODULE_PATH, new=True)
    return patch, installed


def run(root: Path, runtime: str, action: str) -> str:
    root = root.resolve()
    if Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("--vllm-dir must name the git checkout root")
    if git(root, "rev-parse", "HEAD") != PIN:
        raise ValueError(f"expected vLLM HEAD {PIN}; no checkout/reset performed")
    patch, installed = prepare(root, runtime)
    if action == "reverse" and not installed:
        return "Already absent; nothing changed."
    if action == "apply" and installed:
        return "Already installed with identical files; nothing changed."
    reverse = installed if action == "check" else action == "reverse"
    flags = ["-R"] if reverse else []
    git(root, "apply", *flags, "--check", "--whitespace=error", "-", data=patch)
    if action == "check":
        return ("Installed and reversibility checked; nothing changed." if installed else
                "Forward patch checked; nothing changed. Use --apply explicitly.")
    git(root, "apply", *flags, "--whitespace=error", "-", data=patch)
    return ("Experiment removed. Original PLE patch and launchers unchanged." if reverse else
            "Experiment installed, OFF by default. Activation log is required before benchmarking.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-dir", type=Path, required=True)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check", dest="action", action="store_const", const="check")
    actions.add_argument("--apply", dest="action", action="store_const", const="apply")
    actions.add_argument("--reverse", dest="action", action="store_const", const="reverse")
    parser.set_defaults(action="check")
    args = parser.parse_args()
    runtime = (Path(__file__).resolve().parents[1] / "experiments" /
               "strata_draft_vocab.py").read_text(encoding="utf-8")
    try:
        print(run(args.vllm_dir, runtime, args.action))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
