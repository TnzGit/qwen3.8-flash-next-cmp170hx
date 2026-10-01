# SPDX-License-Identifier: Apache-2.0
"""Experimental, opt-in Qwen4Exp MTP head subset. Target logits are untouched.

Independent implementation inspired by Strata's draft-only vocabulary reduction.
CLI: build a local-tokenizer-specific manifest, or run a synthetic CUDA head bench.
The installer copies this module into the pinned vLLM checkout; it is not a plugin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch
from torch import nn
from torch.nn import functional as F

KEY = "qwen4_mtp_draft_vocab"
SHA_KEY = KEY + "_sha256"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_ids(ids: object, vocab_size: int) -> list[int]:
    if not isinstance(ids, list) or not ids:
        raise ValueError("token_ids must be a nonempty list")
    if any(type(i) is not int or not 0 <= i < vocab_size for i in ids):
        raise ValueError("token_ids must be integer IDs in the target vocabulary")
    if ids != sorted(set(ids)):
        raise ValueError("token_ids must be unique and sorted")
    return ids


def read_manifest(path: Path, vocab_size: int, expected_sha: str,
                  tokenizer_path: Path) -> list[int]:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("manifest exceeds 16 MiB")
    raw = path.read_bytes()
    if not expected_sha or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("draft vocabulary SHA256 mismatch or missing config hash")
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("unsupported draft vocabulary manifest")
    if type(data.get("vocab_size")) is not int or data["vocab_size"] != vocab_size:
        raise ValueError("target vocabulary size mismatch")
    if data.get("tokenizer_sha256") != digest(tokenizer_path):
        raise ValueError("tokenizer.json SHA256 mismatch")
    return validate_ids(data.get("token_ids"), vocab_size)


class SubsetLogitsProcessor(nn.Module):
    """Keep the existing processor, project selected rows, scatter to full IDs.

    Build AFTER vLLM shares the target LM head, BEFORE graph capture. The target's
    head and processor are not modified. The additional BF16 weight copy is real
    VRAM overhead, not a memory saving. No CPU work/copies occur in forward.
    """
    def __init__(self, processor: nn.Module, head: nn.Module,
                 ids: list[int], vocab_size: int):
        super().__init__()
        validate_ids(ids, vocab_size)
        if head.weight.ndim != 2 or head.weight.shape[0] < vocab_size:
            raise ValueError("head shape does not cover target vocabulary")
        self.original = processor
        self.vocab_size = vocab_size
        self._source_weight_id = id(head.weight)
        self.register_buffer("ids", torch.tensor(ids, dtype=torch.long,
                                                  device=head.weight.device),
                             persistent=False)
        self.head = nn.Module()
        self.head.tp_size = 1
        self.head.quant_method = head.quant_method
        self.head.register_buffer("weight", head.weight.detach().index_select(
            0, self.ids).contiguous(), persistent=False)

    def forward(self, head: nn.Module, hidden_states: torch.Tensor,
                embedding_bias=None, skip_gather: bool = False) -> torch.Tensor:
        if id(head.weight) != self._source_weight_id:
            raise RuntimeError("LM head rebound after subset preparation; restart")
        if embedding_bias is not None or skip_gather:
            raise ValueError("subset prototype requires default unbiased logits path")
        # Applying soft-cap/scale BEFORE scattering leaves excluded IDs at -inf.
        compact = self.original(self.head, hidden_states)
        logits = compact.new_full((*compact.shape[:-1], self.vocab_size), -math.inf)
        return logits.index_copy_(-1, self.ids, compact)


def validate_settings(config) -> None:
    spec = config.speculative_config
    if spec is None or spec.method != "mtp" or spec.draft_sample_method != "greedy":
        raise ValueError("prototype requires MTP with greedy draft sampling")
    if spec.use_local_argmax_reduction or spec.enable_adaptive_verification:
        raise ValueError("disable local argmax reduction/adaptive verification for this test")
    for field in ("tensor_parallel_size", "pipeline_parallel_size",
                  "data_parallel_size", "prefill_context_parallel_size",
                  "decode_context_parallel_size"):
        if getattr(config.parallel_config, field, 1) != 1:
            raise ValueError(f"prototype requires {field}=1")
    if config.lora_config is not None or getattr(config, "watermark_config", None) is not None:
        raise ValueError("LoRA and watermarking are outside this prototype")


def install(model: nn.Module, config) -> bool:
    """Called only by the new-runner MTP load hook, after load_eagle_model()."""
    path = config.additional_config.get(KEY)
    if not path:
        return False
    validate_settings(config)
    from vllm.models.qwen4_exp.nvidia.mtp import Qwen4ExpMTP
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
    from vllm.logger import init_logger
    if not isinstance(model, Qwen4ExpMTP):
        raise ValueError("only the NVIDIA Qwen4ExpMTP implementation is supported")
    head = model.lm_head
    processor = model.logits_processor
    if isinstance(processor, SubsetLogitsProcessor):
        raise ValueError("subset processor already installed")
    if type(head.quant_method) not in (UnquantizedEmbeddingMethod, UnquantizedLinearMethod):
        raise ValueError("requires the unchanged unquantized BF16 head method")
    if head.tp_size != 1 or head.weight.dtype != torch.bfloat16:
        raise ValueError("requires a TP=1 BF16 LM head")
    if not head.weight.is_cuda or torch.cuda.get_device_capability(head.weight.device) != (8, 0):
        raise ValueError("runtime prototype is scoped to CUDA SM80")
    if processor.logits_as_input or processor.head_dtype not in (None, torch.bfloat16):
        raise ValueError("requires the default BF16 projection path")
    model_dir = Path(config.model_config.tokenizer or config.model_config.model)
    ids = read_manifest(Path(path), model.config.vocab_size,
                        config.additional_config.get(SHA_KEY, ""),
                        model_dir / "tokenizer.json")
    wrapped = SubsetLogitsProcessor(processor, head, ids, model.config.vocab_size)
    model.logits_processor = wrapped
    init_logger(__name__).warning(
        "EXPERIMENTAL qwen4 MTP draft vocabulary ACTIVE: %d/%d rows; extra weight %.1f MiB; sha256=%s",
        len(ids), model.config.vocab_size,
        wrapped.head.weight.numel() * wrapped.head.weight.element_size() / 2**20,
        config.additional_config[SHA_KEY])
    return True


def is_cjk(text: str) -> bool:
    ranges = ((0x1100, 0x11FF), (0x2E80, 0xA4CF), (0xA960, 0xA97F),
              (0xAC00, 0xD7FF), (0xF900, 0xFAFF), (0xFE30, 0xFE4F),
              (0xFF00, 0xFFEF), (0x1AFF0, 0x1B16F), (0x20000, 0x323AF))
    return any(a <= ord(c) <= b for c in text for a, b in ranges)


def geometry(model_dir: Path) -> tuple[int, int]:
    cfg = json.loads((model_dir / "config.json").read_text())
    cfg = cfg.get("text_config", cfg)
    return int(cfg["vocab_size"]), int(cfg["hidden_size"])


def build(args) -> None:
    model_dir = Path(args.model_dir)
    vocab_size, hidden_size = geometry(model_dir)
    tok_path = model_dir / "tokenizer.json"
    if not 0 <= args.base_ids <= vocab_size:
        raise ValueError("base-ids must lie within the vocabulary")
    if args.mode == "full":
        ids = list(range(vocab_size))
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(tok_path))
        available = sorted(set(tok.get_vocab().values()))
        validate_ids(available, vocab_size)
        raw_tok = json.loads(tok_path.read_text())
        keep = {t["id"] for t in raw_tok.get("added_tokens", []) if t.get("special")}
        for i in available:
            text = tok.decode([i], skip_special_tokens=False)
            # Low ID is a conservative heuristic, NOT a frequency measurement.
            # Keep byte fragments too; a single-token decode can be incomplete.
            if i < args.base_ids or text.isascii() or "\ufffd" in text or is_cjk(text):
                keep.add(i)
        for corpus in args.corpus:
            if Path(corpus).stat().st_size > 64 * 1024 * 1024:
                raise ValueError("each calibration corpus must be <=64 MiB")
            with open(corpus, encoding="utf-8") as f:
                for line in f:
                    keep.update(tok.encode(line, add_special_tokens=False).ids)
        ids = sorted(keep)
    validate_ids(ids, vocab_size)
    manifest = {"schema_version": 1, "vocab_size": vocab_size,
                "tokenizer_sha256": digest(tok_path), "selection": args.mode,
                "base_ids": args.base_ids, "token_ids": ids}
    out = Path(args.out).resolve()
    with out.open("x", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    print(json.dumps({"rows": len(ids), "vocab_size": vocab_size,
                      "extra_bf16_weight_mib": len(ids) * hidden_size * 2 / 2**20,
                      "additional_config_to_MERGE": {KEY: str(out), SHA_KEY: digest(out)}},
                     indent=2))


class _LinearMethod:
    def apply(self, head, x, bias=None):
        return F.linear(x, head.weight, bias)


class _LinearProcessor(nn.Module):
    def forward(self, head, x):
        return head.quant_method.apply(head, x)


@torch.inference_mode()
def bench(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; this benchmark has not run on CPU")
    model_dir = Path(args.model_dir)
    v, h = geometry(model_dir)
    path = Path(args.manifest)
    ids = read_manifest(path, v, digest(path), model_dir / "tokenizer.json")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(17)
    head = nn.Module()
    head.quant_method = _LinearMethod()
    head.register_buffer("weight", torch.empty((v, h), device=device,
                         dtype=torch.bfloat16).normal_(std=1 / math.sqrt(h)))
    processor = _LinearProcessor()
    subset = SubsetLogitsProcessor(processor, head, ids, v)
    results = []
    for rows in args.rows:
        x = torch.randn((rows, h), device=device, dtype=torch.bfloat16)
        fns = {"full": lambda: processor(head, x), "subset": lambda: subset(head, x)}
        graphs = {}
        for name, fn in fns.items():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(10):
                    fn()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = fn()
            graphs[name] = (graph, output)
        samples = {name: [] for name in graphs}
        for repeat in range(6):
            order = ("full", "subset") if repeat % 2 == 0 else ("subset", "full")
            for name in order:
                graph = graphs[name][0]
                for _ in range(10):
                    graph.replay()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iters):
                    graph.replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) / args.iters)
        full_ms, sub_ms = (statistics.median(samples[n]) for n in ("full", "subset"))
        results.append({"rows": rows, "full_ms": full_ms, "subset_ms_including_scatter": sub_ms,
                        "ratio": full_ms / sub_ms, "samples_ms": samples,
                        "saved_ms_per_3_draft_heads": 3 * (full_ms - sub_ms)})
    print(json.dumps({"kind": "SYNTHETIC_HEAD_ONLY_NOT_END_TO_END", "torch": torch.__version__,
                      "gpu": torch.cuda.get_device_name(device), "capability": torch.cuda.get_device_capability(device),
                      "shape": [v, h], "subset_rows": len(ids), "results": results}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--mode", choices=("full", "cjk-code"), required=True)
    p.add_argument("--base-ids", type=int, default=65536)
    p.add_argument("--corpus", nargs="*", default=[])
    p.add_argument("--out", required=True)
    p.set_defaults(run=build)
    p = sub.add_parser("bench")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--rows", type=int, nargs="+", default=[1, 4])
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--device", type=int, default=0)
    p.set_defaults(run=bench)
    args = parser.parse_args()
    if args.command == "bench" and (args.iters < 1 or min(args.rows) < 1):
        parser.error("rows and iters must be positive")
    args.run(args)


if __name__ == "__main__":
    main()
