# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests and an optional CUDA graph test; no vLLM/model download."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp = load("strata_draft_vocab", ROOT / "experiments/strata_draft_vocab.py")
installer = load("strata_installer", ROOT / "scripts/apply-strata-draft-vocab.py")

# Audited upstream file, used ONLY as a patch-application fixture. This is not a
# running vLLM checkout. Its Git blob hash must match the pinned source exactly.
ORIGINAL = '''# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model


class MTPSpeculator(AutoRegressiveSpeculator):
    share_mtp_topk_indices: bool = False

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        draft_model = load_eagle_model(target_model, self.vllm_config)
        spec_config = self.vllm_config.speculative_config
        draft_hf_config = (
            spec_config.draft_model_config.hf_config
            if spec_config is not None
            else None
        )
        # Detect index_share_for_mtp_iteration. When True, the proposer
        # toggles skip_topk so step 0 computes MTP's own indices and
        # steps 1+ reuse them.
        self.share_mtp_topk_indices = (
            self.vllm_config.parallel_config.prefill_context_parallel_size == 1
            and getattr(draft_hf_config, "index_share_for_mtp_iteration", False)
            and hasattr(draft_model.model, "set_skip_topk")
            and hasattr(draft_model.model, "compact_topk_indices")
        )
        return draft_model

    def on_prefill_begin(self, num_reqs: int) -> None:
        # Step 0 computes its own top-k. Unconditional, so a step that died
        # midway cannot leave reuse mode on.
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(False)

    def on_prefill_end(self, num_reqs: int) -> None:
        # Step 0 (prefill) wrote topk indices for every query token in the
        # multi-token batch. Compact them down to each request's last token so
        # steps 1+ can reuse them from the shared buffer.
        if self.share_mtp_topk_indices and self.num_speculative_steps > 1:
            self.model.model.compact_topk_indices(self.last_token_indices[:num_reqs])

    def on_multi_step_decode_begin(self, num_reqs: int) -> None:
        # Switch to reuse mode so draft steps 1+ skip the indexer op and read
        # the indices that step 0 wrote into the shared buffer.
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(True)

    def on_multi_step_decode_end(self, num_reqs: int) -> None:
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(False)
'''

# Reduced legacy-proposer fixture: only loader/share ordering is reproduced.
# The production installer checks the FULL upstream legacy file's Git blob;
# this fixture substitutes its own checksum only via pytest monkeypatch.
LEGACY_ORIGINAL = """class SpecDecodeBaseProposer:
    def load_model(self, target_language_model):
        self._maybe_share_lm_head(target_language_model)

    def _maybe_share_lm_head(self, target_language_model):
        self.model.lm_head = target_language_model.lm_head
"""


def settings():
    return NS(speculative_config=NS(method="mtp", draft_sample_method="greedy",
                                   use_local_argmax_reduction=False,
                                   enable_adaptive_verification=False),
              parallel_config=NS(), lora_config=None, watermark_config=None,
              additional_config={})


def head_and_processor(device="cpu", dtype=torch.float32):
    head = nn.Module()
    head.quant_method = exp._LinearMethod()
    head.register_buffer("weight", torch.arange(64, device=device, dtype=dtype).reshape(16, 4) / 64)
    return head, exp._LinearProcessor()


@pytest.mark.parametrize("ids", [[], [True], [1.0], [-1], [16], [3, 3], [4, 2], "1,2"])
def test_invalid_ids(ids):
    with pytest.raises(ValueError):
        exp.validate_ids(ids, 16)


@pytest.mark.parametrize("rows", [1, 4, 7])
def test_logits_global_ids_and_target_unchanged(rows):
    head, processor = head_and_processor()
    before = head.weight.clone()
    wrapped = exp.SubsetLogitsProcessor(processor, head, [0, 3, 11, 15], 16)
    x = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4) / 32
    full, actual = processor(head, x), wrapped(head, x)
    torch.testing.assert_close(actual[:, [0, 3, 11, 15]], full[:, [0, 3, 11, 15]])
    assert torch.isneginf(actual[:, [1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]]).all()
    assert actual.shape == (rows, 16)
    assert torch.all(actual.argmax(-1) == 15)  # global ID, not compact index 3
    assert wrapped.head.weight.data_ptr() != head.weight.data_ptr()
    torch.testing.assert_close(head.weight, before, rtol=0, atol=0)
    torch.testing.assert_close(processor(head, x), full, rtol=0, atol=0)


def test_full_vocab_control():
    head, processor = head_and_processor()
    wrapped = exp.SubsetLogitsProcessor(processor, head, list(range(16)), 16)
    x = torch.ones(3, 4)
    torch.testing.assert_close(wrapped(head, x), processor(head, x), rtol=0, atol=0)


def test_scatter_after_softcap():
    class Softcap(nn.Module):
        def forward(self, head, x):
            return torch.tanh(head.quant_method.apply(head, x) / 2) * 6
    head, _ = head_and_processor()
    processor = Softcap()
    wrapped = exp.SubsetLogitsProcessor(processor, head, [1, 5], 16)
    x = torch.ones(2, 4)
    out = wrapped(head, x)
    torch.testing.assert_close(out[:, [1, 5]], processor(head, x)[:, [1, 5]])
    assert torch.isneginf(out[:, 0]).all()


def test_unsupported_logits_paths():
    head, processor = head_and_processor()
    wrapped = exp.SubsetLogitsProcessor(processor, head, [1], 16)
    x = torch.ones(1, 4)
    with pytest.raises(ValueError, match="unbiased"):
        wrapped(head, x, embedding_bias=torch.zeros(16))
    with pytest.raises(ValueError, match="unbiased"):
        wrapped(head, x, skip_gather=True)
    head.weight = head.weight.clone()
    with pytest.raises(RuntimeError, match="rebound"):
        wrapped(head, x)


def test_default_off_does_not_import_vllm():
    assert exp.install(nn.Module(), NS(additional_config={})) is False


@pytest.mark.parametrize("field,value", [("method", "eagle"), ("draft_sample_method", "probabilistic"),
                                        ("use_local_argmax_reduction", True),
                                        ("enable_adaptive_verification", True)])
def test_spec_guards(field, value):
    cfg = settings()
    setattr(cfg.speculative_config, field, value)
    with pytest.raises(ValueError):
        exp.validate_settings(cfg)


@pytest.mark.parametrize("field", ["tensor_parallel_size", "pipeline_parallel_size",
                                  "data_parallel_size", "prefill_context_parallel_size",
                                  "decode_context_parallel_size"])
def test_parallel_guards(field):
    cfg = settings()
    setattr(cfg.parallel_config, field, 2)
    with pytest.raises(ValueError, match=field):
        exp.validate_settings(cfg)


@pytest.mark.parametrize("field", ["lora_config", "watermark_config"])
def test_adapter_guards(field):
    cfg = settings()
    setattr(cfg, field, NS())
    with pytest.raises(ValueError):
        exp.validate_settings(cfg)


def test_supported_settings():
    exp.validate_settings(settings())


def manifest_fixture(tmp_path):
    tok = tmp_path / "tokenizer.json"
    tok.write_text('{}\n')
    manifest = tmp_path / "vocab.json"
    manifest.write_text(json.dumps({"schema_version": 1, "vocab_size": 16,
                                   "tokenizer_sha256": exp.digest(tok),
                                   "token_ids": [0, 3, 15]}))
    return manifest, tok


def test_manifest_hash_and_tokenizer_binding(tmp_path):
    path, tok = manifest_fixture(tmp_path)
    sha = exp.digest(path)
    assert exp.read_manifest(path, 16, sha, tok) == [0, 3, 15]
    with pytest.raises(ValueError, match="vocabulary size"):
        exp.read_manifest(path, 17, sha, tok)
    with pytest.raises(ValueError, match="SHA256"):
        exp.read_manifest(path, 16, "", tok)
    tok.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="tokenizer"):
        exp.read_manifest(path, 16, sha, tok)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="SHA256"):
        exp.read_manifest(path, 16, sha, tok)


def test_build_full_is_exclusive(tmp_path, capsys):
    (tmp_path / "config.json").write_text('{"text_config":{"vocab_size":16,"hidden_size":4}}')
    (tmp_path / "tokenizer.json").write_text('{}')
    args = NS(model_dir=str(tmp_path), mode="full", base_ids=0, corpus=[], out=str(tmp_path / "map.json"))
    exp.build(args)
    out = json.loads(capsys.readouterr().out)
    assert out["rows"] == 16
    conf = out["additional_config_to_MERGE"]
    assert exp.read_manifest(Path(conf[exp.KEY]), 16, conf[exp.SHA_KEY], tmp_path / "tokenizer.json") == list(range(16))
    with pytest.raises(FileExistsError):
        exp.build(args)


def test_build_cjk_code_heuristic_mock_tokenizer(tmp_path, monkeypatch):
    # Selection policy only; actual local model tokenizer remains a GPU-host check.
    texts = ["a", "汉", "あ", "가", "русский", "\ufffd", "<special>", "Ω", "слово"]
    class Tokenizer:
        @staticmethod
        def from_file(path):
            return Tokenizer()
        def get_vocab(self):
            return {text: i for i, text in enumerate(texts)}
        def decode(self, ids, **kwargs):
            return texts[ids[0]]
        def encode(self, line, **kwargs):
            return NS(ids=[7])
    monkeypatch.setitem(sys.modules, "tokenizers", NS(Tokenizer=Tokenizer))
    (tmp_path / "config.json").write_text('{"vocab_size":10,"hidden_size":4}')
    (tmp_path / "tokenizer.json").write_text('{"added_tokens":[{"id":6,"special":true}]}')
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Ω\n")
    out = tmp_path / "map.json"
    exp.build(NS(model_dir=str(tmp_path), mode="cjk-code", base_ids=0,
                 corpus=[str(corpus)], out=str(out)))
    assert json.loads(out.read_text())["token_ids"] == [0, 1, 2, 3, 5, 6, 7]


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "vllm"
    source = root / installer.HOOK_PATH
    source.parent.mkdir(parents=True)
    (root / installer.MODULE_PATH).parent.mkdir(parents=True)
    source.write_text(ORIGINAL)
    legacy = root / installer.LEGACY_PATH
    legacy.parent.mkdir(parents=True)
    legacy.write_text(LEGACY_ORIGINAL)
    monkeypatch.setattr(installer, "LEGACY_BLOB", installer.blob_sha(LEGACY_ORIGINAL.encode()))
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    installer.git(root, "add", ".")
    installer.git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                  "commit", "-qm", "test fixture only")
    return root


def test_fixture_matches_audited_git_blob():
    assert installer.blob_sha(ORIGINAL.encode()) == installer.SOURCE_BLOB


def test_installer_forward_reverse_and_idempotency(checkout, monkeypatch):
    # Production pin is NOT bypassable via CLI. Test fixture needs its own HEAD.
    monkeypatch.setattr(installer, "PIN", installer.git(checkout, "rev-parse", "HEAD"))
    runtime = (ROOT / "experiments/strata_draft_vocab.py").read_text()
    assert "nothing changed" in installer.run(checkout, runtime, "check")
    assert installer.git(checkout, "status", "--porcelain") == ""
    assert "OFF by default" in installer.run(checkout, runtime, "apply")
    actual = (checkout / installer.HOOK_PATH).read_text()
    assert installer.ANCHOR + installer.HOOK in actual
    assert installer.LEGACY_ANCHOR + installer.LEGACY_HOOK in (checkout / installer.LEGACY_PATH).read_text()
    assert (checkout / installer.MODULE_PATH).read_text() == runtime
    assert "Already installed" in installer.run(checkout, runtime, "apply")
    assert "reversibility checked" in installer.run(checkout, runtime, "check")
    assert "removed" in installer.run(checkout, runtime, "reverse")
    assert (checkout / installer.HOOK_PATH).read_text() == ORIGINAL
    assert (checkout / installer.LEGACY_PATH).read_text() == LEGACY_ORIGINAL
    assert not (checkout / installer.MODULE_PATH).exists()
    assert installer.git(checkout, "status", "--porcelain") == ""
    assert "Already absent" in installer.run(checkout, runtime, "reverse")


def test_wrong_head_refused(checkout):
    with pytest.raises(ValueError, match="HEAD"):
        installer.run(checkout, "# runtime\n", "apply")
    assert installer.git(checkout, "status", "--porcelain") == ""


def test_foreign_changes_refused(checkout, monkeypatch):
    monkeypatch.setattr(installer, "PIN", installer.git(checkout, "rev-parse", "HEAD"))
    source = checkout / installer.HOOK_PATH
    source.write_text(ORIGINAL + "# local customization\n")
    with pytest.raises(ValueError, match="audited"):
        installer.run(checkout, "# runtime\n", "apply")
    assert source.read_text().endswith("# local customization\n")


def test_modified_installed_module_not_removed(checkout, monkeypatch):
    monkeypatch.setattr(installer, "PIN", installer.git(checkout, "rev-parse", "HEAD"))
    runtime = "# runtime\n"
    installer.run(checkout, runtime, "apply")
    module = checkout / installer.MODULE_PATH
    module.write_text("# user edit\n")
    with pytest.raises(ValueError, match="modified"):
        installer.run(checkout, runtime, "reverse")
    assert module.read_text() == "# user edit\n"


def test_legacy_edit_refused(checkout, monkeypatch):
    monkeypatch.setattr(installer, "PIN", installer.git(checkout, "rev-parse", "HEAD"))
    path = checkout / installer.LEGACY_PATH
    path.write_text(LEGACY_ORIGINAL + "# user customization\n")
    with pytest.raises(ValueError, match="audited"):
        installer.run(checkout, "# runtime\n", "apply")
    assert path.read_text().endswith("# user customization\n")


def test_partial_install_refused(checkout, monkeypatch):
    monkeypatch.setattr(installer, "PIN", installer.git(checkout, "rev-parse", "HEAD"))
    path = checkout / installer.HOOK_PATH
    path.write_text(ORIGINAL.replace(installer.ANCHOR, installer.ANCHOR + installer.HOOK))
    with pytest.raises(ValueError, match="partially installed"):
        installer.run(checkout, "# runtime\n", "apply")


@pytest.mark.parametrize("enabled", [False, True])
def test_legacy_hook_runs_after_sharing(checkout, monkeypatch, enabled):
    from types import ModuleType
    monkeypatch.setattr(installer, "PIN", installer.git(checkout, "rev-parse", "HEAD"))
    installer.run(checkout, "# runtime\n", "apply")
    seen = []
    module = ModuleType("vllm.models.qwen4_exp.nvidia.mtp_draft_vocab")
    module.install = lambda model, config: seen.append(model.lm_head)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace = {}
    exec((checkout / installer.LEGACY_PATH).read_text(), namespace)
    proposer = namespace["SpecDecodeBaseProposer"]()
    proposer.model = NS(lm_head=object())
    proposer.vllm_config = NS(additional_config={exp.KEY: "map.json"} if enabled else {})
    target = NS(lm_head=object())
    proposer.load_model(target)
    assert proposer.model.lm_head is target.lm_head
    assert seen == ([target.lm_head] if enabled else [])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires local CUDA GPU")
def test_cuda_graph_replay_uses_new_inputs():
    head, processor = head_and_processor("cuda", torch.bfloat16)
    wrapped = exp.SubsetLogitsProcessor(processor, head, [0, 3, 15], 16)
    x = torch.ones(1, 4, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            wrapped(head, x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = wrapped(head, x)
    for multiplier in (1, 2, 4):
        x.fill_(multiplier)
        graph.replay()
        torch.cuda.synchronize()
        ref = processor(head, x)
        torch.testing.assert_close(out[:, [0, 3, 15]], ref[:, [0, 3, 15]])
        assert torch.isneginf(out[:, 1]).all()
