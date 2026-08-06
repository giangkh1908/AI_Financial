"""Test config.py — deep merge base+override, extra='forbid' (fix #3, #9)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from vifinqa.config import Config

ROOT = Path(__file__).resolve().parents[1]


def test_load_merges_base_with_override(tmp_path: Path):
    # base set một field KHÔNG có trong override → phải giữ giá trị base
    base = tmp_path / "base.yaml"
    base.write_text(
        "paths:\n  data_dir: base_data\nretrieval:\n  k: 50\nllm:\n  max_tokens: 999\n",
        encoding="utf-8",
    )
    override = tmp_path / "api.yaml"
    override.write_text(
        "llm:\n  model_id: test-model\n",
        encoding="utf-8",
    )
    cfg = Config.load(override, base_path=base)
    assert cfg.paths["data_dir"] == "base_data"          # từ base
    assert cfg.retrieval.k == 50                         # từ base
    assert cfg.llm.max_tokens == 999                     # từ base
    assert cfg.llm.model_id == "test-model"              # override ghi đè


def test_load_override_wins(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text("step_budget: 10\n", encoding="utf-8")
    override = tmp_path / "api.yaml"
    override.write_text("step_budget: 42\n", encoding="utf-8")
    cfg = Config.load(override, base_path=base)
    assert cfg.step_budget == 42


def test_config_extra_forbid(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text("paths:\n  data_dir: data\n", encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text("llm:\n  max_token: 5\n", encoding="utf-8")  # typo
    with pytest.raises(ValidationError):
        Config.load(bad)


def test_from_yaml_unknown_top_level_key_raises(tmp_path: Path):
    f = tmp_path / "x.yaml"
    f.write_text("retrival_k: 10\n", encoding="utf-8")  # typo
    with pytest.raises(ValidationError):
        Config.from_yaml(f)


def test_retrieval_nested_defaults():
    # M3/M3.1: các sub-config retrieval có giá trị mặc định đúng (base.yaml thật)
    cfg = Config.from_yaml(ROOT / "configs" / "base.yaml")
    r = cfg.retrieval
    assert r.engine == "qdrant"
    assert r.use_dense is True and r.use_sparse is True
    assert r.fusion == "native"
    assert r.statement_bonus == 0.001        # M3.1: giảm từ 0.05 (RRF max ≈0.0328)
    assert r.embedding.model == "baai/bge-m3"
    assert r.embedding.dense_dim == 1024
    assert r.embedding.max_chars == 4000     # M3.1: bump từ 2000
    assert r.embedding.workers == 4         # M3.1: giảm từ 12 (tránh 429)
    assert r.sparse.deep_labels_cap == 0
    assert r.embed_statement_only is True    # M3.1: chỉ 10,797 bảng BCTC
    # qdrant server (Docker localhost) mặc định
    assert r.qdrant.mode == "server"
    assert r.qdrant.host == "localhost" and r.qdrant.port == 6333
    assert r.qdrant.collection == "bctc_tables"
    assert r.qdrant.hnsw_m == 16 and r.qdrant.hnsw_on_disk is True
    # rerank tắt (M3.1)
    assert r.rerank.enabled is False
    assert r.rerank.model == ""               # trống khi disabled
    assert r.rerank.device == "cpu" and r.rerank.candidates == 50


def test_retrieval_override_nested(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "retrieval:\n  embedding:\n    model: baai/bge-m3\n  rerank:\n    device: cpu\n",
        encoding="utf-8",
    )
    override = tmp_path / "api.yaml"
    override.write_text(
        "retrieval:\n  rerank:\n    device: cuda\n  qdrant:\n    hnsw_m: 64\n",
        encoding="utf-8",
    )
    cfg = Config.load(override, base_path=base)
    assert cfg.retrieval.rerank.device == "cuda"      # override
    assert cfg.retrieval.qdrant.hnsw_m == 64           # override
    assert cfg.retrieval.embedding.model == "baai/bge-m3"  # giữ base


def test_retrieval_nested_extra_forbid(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text("retrieval:\n  embedding:\n    model: x\n", encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text("retrieval:\n  embedding:\n    modelx: y\n", encoding="utf-8")  # typo nested
    with pytest.raises(ValidationError):
        Config.load(bad)
