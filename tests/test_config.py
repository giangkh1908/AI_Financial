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
