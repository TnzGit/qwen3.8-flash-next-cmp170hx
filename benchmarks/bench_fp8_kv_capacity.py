"""Cold, independent long-context concurrency probe for Qwen3.8 vLLM.

Use separately for BF16 and FP8 under *identical* model/MTP/graph settings.
This uses random valid token IDs to exercise cache capacity, not language quality.
Only metrics are saved; raw prompts and completions are not written to disk.
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import aiohttp


async def probe(session, url, model, tokens, output_tokens):
    payload = {
        "model": model,
        "prompt": tokens,
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    first = None
    usage = None
    streamed_pieces = 0
    try:
        async with session.post(url.rstrip("/") + "/v1/completions", json=payload) as res:
            if res.status != 200:
                return {
                    "ok": False, "status": res.status,
                    "error": (await res.text())[:1500],
                    "elapsed_s": time.perf_counter() - start,
                }
            async for raw in res.content:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                data = json.loads(line[6:])
                if data.get("usage"):
                    usage = data["usage"]
                for choice in data.get("choices", []):
                    if choice.get("text"):
                        first = first or time.perf_counter()
                        streamed_pieces += 1
        end = time.perf_counter()
        if not usage or first is None:
            return {"ok": False, "error": "Missing streaming token content or usage", "elapsed_s": end - start}
        emitted = int(usage.get("completion_tokens", 0))
        return {
            "ok": True, "ttft_s": round(first - start, 4),
            "elapsed_s": round(end - start, 4),
            "generation_s": round(end - first, 4),
            "decode_tok_s": round(max(emitted - 1, 0) / max(end - first, 1e-6), 3),
            "output_tokens": emitted,
            "prompt_tokens_reported": usage.get("prompt_tokens"),
            "streamed_pieces": streamed_pieces,
        }
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": type(exc).__name__ + ": " + str(exc)[:1000],
                "elapsed_s": round(time.perf_counter() - start, 4)}


async def run(args):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for length in args.lengths:
            for concurrency in args.concurrency:
                for repeat in range(args.rounds):
                    # Use independent deterministic seeds, no shared prompt prefix.
                    prompts = []
                    for i in range(concurrency):
                        rng = random.Random(args.seed + length * 1009 + concurrency * 7919
                                            + repeat * 104729 + i * 65537)
                        prompts.append([rng.randrange(1000, 30000) for _ in range(length)])
                    start = time.perf_counter()
                    responses = await asyncio.gather(*[
                        probe(session, args.url, args.model, prompt, args.output_tokens)
                        for prompt in prompts
                    ])
                    elapsed = time.perf_counter() - start
                    ok = all(r["ok"] for r in responses)
                    if ok and any(r["prompt_tokens_reported"] != length for r in responses):
                        ok = False
                    row = {
                        "label": args.label, "length": length,
                        "concurrency": concurrency, "round": repeat,
                        "output_max_tokens": args.output_tokens,
                        "wall_s": round(elapsed, 3), "all_success": ok,
                        "aggregate_output_tok_s": round(
                            sum(r.get("output_tokens", 0) for r in responses) / max(elapsed, 1e-6), 3
                        ) if ok else None,
                        "median_ttft_s": statistics.median(
                            r["ttft_s"] for r in responses
                        ) if ok else None,
                        "requests": responses,
                    }
                    results.append(row)
                    print(json.dumps({k: v for k, v in row.items() if k != "requests"}), flush=True)
                    output.write_text(json.dumps(results, indent=2) + "\n")
                    if not ok and args.stop_on_error:
                        raise SystemExit("Probe failed; inspect JSON and vLLM metrics/logs.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8003")
    parser.add_argument("--model", default="Qwen3.8-Flash-Next")
    parser.add_argument("--label", required=True, help="Example: bf16-4graph or fp8-4graph")
    parser.add_argument("--output", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[8192, 32768, 131072, 220000, 240000])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=83501)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    if any(n <= 0 for n in args.lengths + args.concurrency) or args.output_tokens <= 0:
        parser.error("Lengths, concurrency and output-tokens must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
