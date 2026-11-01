# SPDX-License-Identifier: Apache-2.0
"""GPU-free regression checks for the pinned vLLM #60602 QSA backport.

These tests do not claim to prove CUDA/Triton compilation or model correctness.
Set VLLM_SOURCE to a patched upstream a5a30471... checkout to check integration.
"""

import ast
import os
import struct
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINNED = "a5a30471ff2bb7f0824f2da10e358af98d304472"


def _fp8_reference(code: int) -> float:
    """Finite e4m3fn values; NaNs are intentionally handled separately."""
    exponent = (code >> 3) & 15
    mantissa = code & 7
    sign = -1.0 if code & 0x80 else 1.0
    if exponent == 0:
        return sign * mantissa * (2.0 ** -9)
    return sign * (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))


def _sm80_fp8_byte_decode(code: int) -> float:
    """Triton's bit-cast path gives fp16(e4m3 * 2**-8); host scales by 256."""
    fp16_bits = ((code & 0x80) << 8) | ((code & 0x7F) << 7)
    return struct.unpack("<e", struct.pack("<H", fp16_bits))[0] * 256.0


class TestByteDecode(unittest.TestCase):
    def test_all_254_finite_fp8_e4m3_codes(self):
        for code in range(256):
            if (code & 0x7F) == 0x7F:
                continue
            with self.subTest(hex=hex(code)):
                self.assertEqual(_sm80_fp8_byte_decode(code), _fp8_reference(code))

    def test_nan_codes_are_480_not_nan(self):
        self.assertEqual(_sm80_fp8_byte_decode(0x7F), 480.0)
        self.assertEqual(_sm80_fp8_byte_decode(0xFF), -480.0)


class TestBackportSource(unittest.TestCase):
    def test_opt_in_defaults_keep_existing_production_profile(self):
        serve = (ROOT / "scripts/serve.sh").read_text()
        profile = (ROOT / "scripts/serve-cmp170hx-profile.sh").read_text()
        self.assertIn('--kv-cache-dtype "${QWEN_KV_CACHE_DTYPE:-auto}"', serve)
        self.assertIn('QWEN_KV_CACHE_DTYPE:-auto', profile)
        self.assertIn('QWEN_SEQS:-1', profile)
        self.assertIn('QWEN_CONTEXT:-243200', profile)

    def test_upstream_pinned_source_if_present(self):
        source = os.environ.get("VLLM_SOURCE")
        if not source:
            self.skipTest("Set VLLM_SOURCE to patched vLLM checkout")
        source = Path(source)
        paths = [
            source / "vllm/models/qwen4_exp/nvidia/ops/qsa.py",
            source / "vllm/models/qwen4_exp/nvidia/qsa.py",
        ]
        ops, qsa = (path.read_text() for path in paths)
        for path, data in zip(paths, (ops, qsa)):
            with self.subTest(path=str(path)):
                ast.parse(data, filename=str(path))
        # These assertions capture the intended SM80 implementation, including
        # the raw-byte path in both dispatch and warmup.
        for fragment in (
            "if tile.dtype == tl.uint8:",
            "_maybe_dequant(keys).to(query.dtype)",
            "values = _maybe_dequant(values)",
            "if is_fp8 and _is_sm80():",
            "if k_cache.dtype == torch.uint8:",
            "softmax_scale *= 256.0",
            "output_scale *= 256.0",
            "cache_dtype = fp8_dtype if is_fp8 else key_cache.dtype",
            "current_platform.has_device_capability(89)",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, ops)
        self.assertNotIn("IS_FP8=is_fp8", ops)
        self.assertIn("if current_platform.has_device_capability(89):", qsa)
        self.assertIn("torch.uint8,", qsa)
        # The pinned September implementation uses the separate cache writer;
        # the October PR's fused-writer gate is not applicable to this tree.
        self.assertNotIn("self.use_fused_qsa_prepare", qsa)


if __name__ == "__main__":
    unittest.main()
