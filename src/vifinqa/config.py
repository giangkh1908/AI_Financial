"""Cấu hình — load từ YAML (configs/) + env var, provider-agnostic."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# D:\GURU (src/vifinqa/config.py → lên 2 cấp: vifinqa → src → GURU)
ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    """Nạp .env (KEY=VALUE, bỏ qua #) vào os.environ nếu chưa có.

    Dependency-free, tránh phụ thuộc python-dotenv.
    """
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()


class LLMConfig(BaseModel):
    """Cấu hình LLM — dùng chung cho OpenRouter / OpenAI-compatible / vLLM.

    provider = openrouter | openai_compatible | vllm
    base_url là gốc OpenAI-compatible (vd https://openrouter.ai/api/v1).
    """

    provider: str = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""          # nếu rỗng → đọc env OPENROUTER_API_KEY
    model_id: str = "qwen/qwen3.5-9b"
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 60.0
    retries: int = 3
    extra_headers: dict[str, str] = Field(default_factory=dict)

    def effective_api_key(self) -> str:
        """Ưu tiên api_key trong config; fallback env var."""
        if self.api_key:
            return self.api_key
        for env_name in ("OPENROUTER_API_KEY", "VIFINQA_API_KEY"):
            val = os.environ.get(env_name)
            if val:
                return val
        return ""


class RetrievalConfig(BaseModel):
    k: int = 10
    rerank_depth: int = 100
    rrf_k: int = 60


class SandboxConfig(BaseModel):
    timeout: int = 20
    max_code_len: int = 4000
    max_ast_nodes: int = 300


class Config(BaseModel):
    """Cấu hình tổng — gộp base + api/local + env override."""

    paths: dict[str, Any] = Field(default_factory=dict)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    answer_abs_tol: float = 0.01
    step_budget: int = 10
    llm: LLMConfig = Field(default_factory=LLMConfig)

    # ---- paths tiện dụng ----
    @property
    def data_dir(self) -> Path:
        return Path(self.paths.get("data_dir", "data"))

    @property
    def derived_dir(self) -> Path:
        return Path(self.paths.get("derived_dir", "data/derived"))

    @property
    def out_dir(self) -> Path:
        return Path(self.paths.get("out_dir", "data/out"))

    def resolved_data_dir(self) -> Path:
        p = self.data_dir
        return p if p.is_absolute() else ROOT / p

    def resolved_derived_dir(self) -> Path:
        p = self.derived_dir
        return p if p.is_absolute() else ROOT / p

    def resolved_out_dir(self) -> Path:
        p = self.out_dir
        return p if p.is_absolute() else ROOT / p

    # ---- helpers ----
    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load từ 1 file YAML."""
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(**raw)

    @classmethod
    def load(cls, config_path: Path | None = None) -> "Config":
        """Load base.yaml rồi chồng config_path (vd api.yaml) lên."""
        base_path = config_path
        if base_path is None:
            base_path = ROOT / "configs" / "api.yaml"
        base = cls.from_yaml(ROOT / "configs" / "base.yaml")
        extra = cls.from_yaml(base_path)
        # gộp: extra ghi đè base theo field
        merged = base.model_copy(deep=True)
        return merged.model_validate(
            extra.model_dump(exclude_unset=True),
        )
