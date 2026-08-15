"""Cấu hình — load từ YAML (configs/) + env var, provider-agnostic."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

# D:\GURU (src/vifinqa/config.py → lên 2 cấp: vifinqa → src → GURU)
ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: dict, override: dict) -> dict:
    """Gộp đệ quy: override ghi đè base theo từng key (dict lồng nhau merge sâu)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


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


def _env_api_key(provider: str, api_key: str) -> str:
    """api_key từ config; nếu rỗng → resolve env theo provider.

    Dùng chung cho LLMConfig / ProviderRef / EmbedProviderRef. vllm/ollama trả
    placeholder non-empty (OpenAI SDK yêu cầu key khác rỗng dù server không check).
    """
    if api_key:
        return api_key
    if provider == "deepinfra":
        for env_name in ("DEEPINFRA_TOKEN", "DEEPINFRA_API_KEY"):
            val = os.environ.get(env_name)
            if val:
                return val
        return ""
    if provider == "ollama":
        return "ollama"
    if provider == "vllm":
        return os.environ.get("VLLM_API_KEY") or "vllm"
    # openrouter | openai_compatible | http_bge (http_bge không dùng SDK → key không quan trọng)
    for env_name in ("OPENROUTER_API_KEY", "VIFINQA_API_KEY"):
        val = os.environ.get(env_name)
        if val:
            return val
    return ""


class ProviderRef(BaseModel):
    """Một endpoint OpenAI-compatible dùng làm fallback cho LLM.

    provider = vllm | deepinfra | openai_compatible | ollama | openrouter
    Kế thừa temperature/max_tokens/timeout/thinking từ LLMConfig primary (trừ khi ghi đè
    ở đây). model_id bắt buộc vì mỗi provider có thể đặt tên model khác nhau.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str = "openai_compatible"
    base_url: str = ""
    api_key: str = ""          # rỗng → env theo provider
    model_id: str = ""
    timeout: float | None = None      # None → kế thừa primary
    extra_headers: dict[str, str] = Field(default_factory=dict)
    thinking: bool | None = None       # None → kế thừa primary

    def effective_api_key(self) -> str:
        return _env_api_key(self.provider, self.api_key)


class EmbedProviderRef(BaseModel):
    """Một endpoint embed dùng làm fallback (sau primary EmbeddingConfig).

    provider = ngrok | http_bge | deepinfra | openai_compatible | openrouter
    - ngrok/http_bge: HTTP `/embed` tới bge_m3_server (không cần key, không dùng SDK).
    - deepinfra/openai_compatible/openrouter: OpenAI-compatible embeddings API.
    dense_dim phải khớp primary (Embedder assert) — tránh embed sai dim làm hỏng index.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str = "openai_compatible"
    base_url: str = ""
    api_key: str = ""          # rỗng → env theo provider
    model: str = ""
    dense_dim: int = 1024

    def effective_api_key(self) -> str:
        return _env_api_key(self.provider, self.api_key)


class LLMConfig(BaseModel):
    """Cấu hình LLM — dùng chung cho OpenRouter / OpenAI-compatible / vLLM.

    provider = openrouter | openai_compatible | vllm | deepinfra | ollama
    base_url là gốc OpenAI-compatible (vd https://openrouter.ai/api/v1).
    fallbacks = list ProviderRef (thử sau primary; transient → nhảy).
    """

    # extra='forbid': key sai chính tả trong YAML phải raise, không bỏ im lặng
    model_config = ConfigDict(extra="forbid")

    provider: str = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""          # nếu rỗng → đọc env OPENROUTER_API_KEY
    model_id: str = "qwen/qwen3.5-9b"
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 60.0
    retries: int = 3
    extra_headers: dict[str, str] = Field(default_factory=dict)
    thinking: bool = False  # Qwen3 reasoning: tắt mặc định cho codegen (tránh overflow max_tokens + rẻ)
    fallbacks: list[ProviderRef] = Field(default_factory=list)  # thử sau primary khi primary down

    def effective_api_key(self) -> str:
        """Ưu tiên api_key trong config; fallback env var theo provider."""
        return _env_api_key(self.provider, self.api_key)


class EmbeddingConfig(BaseModel):
    """Cấu hình embedding dense — baai/bge-m3 qua OpenRouter Embeddings API.

    Reuse LLMConfig base_url/api_key/extra_headers (cùng OpenRouter key).
    """

    model_config = ConfigDict(extra="forbid")

    provider: str = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"   # endpoint embed; ngrok/http_bge = URL bge_m3_server
    api_key: str = ""          # rỗng → env theo provider (giống LLMConfig)
    model: str = "baai/bge-m3"
    dense_dim: int = 1024
    max_chars: int = 2000            # cap text_dense (≈ 512 tokens bge-m3)
    batch_size: int = 100            # texts / 1 API call
    workers: int = 12                # số API calls chạy song song (OpenAI client thread-safe)
    cache_dir: str = "data/derived/embeddings"   # cache .npy per ticker → resume không trả phí lại
    fallbacks: list[EmbedProviderRef] = Field(default_factory=list)  # thử sau primary khi primary down

    def effective_api_key(self) -> str:
        """Ưu tiên api_key trong config; fallback env theo provider (giống LLMConfig)."""
        return _env_api_key(self.provider, self.api_key)


class SparseConfig(BaseModel):
    """Cấu hình sparse channel — TF/BM25 weights cục bộ → Qdrant sparse vectors."""

    model_config = ConfigDict(extra="forbid")

    max_chars: int = 6000            # cap text_lex (TF sparse)
    deep_labels_cap: int = 0         # 0 = off (M3); N>0 = đọc wide CSV, mở rộng labels đến N


class QdrantConfig(BaseModel):
    """Cấu hình Qdrant local — dense (HNSW) + sparse vectors."""

    model_config = ConfigDict(extra="forbid")

    path: str = "data/derived/qdrant"
    collection: str = "bctc_tables"
    # M3.1: mặc định server (Docker localhost); 'local' = embedded path legacy (single-process, file lock)
    mode: str = "server"
    host: str = "localhost"
    port: int = 6333
    # dense index (HNSW) — tường minh
    hnsw_m: int = 16
    hnsw_ef_construct: int = 128
    hnsw_on_disk: bool = True
    quantize: bool = True            # INT8 scalar quantization dense
    sparse_on_disk: bool = True
    sparse_modifier: str = "idf"     # idf | none


class RerankConfig(BaseModel):
    """Cấu hình rerank — Qwen3-Reranker-0.6B chạy local.

    `model` = đường dẫn local (vd `models/Qwen3-Reranker-0.6B`) hoặc HF hub id
    nếu muốn để transformers tự tải (không khuyến khích — user tự tải vào `models/`).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    provider: str = "local"
    model: str = "models/Qwen3-Reranker-0.6B"
    device: str = "cpu"
    candidates: int = 50             # top-N hybrid đưa vào rerank
    top_k: int = 10
    max_chars_per_doc: int = 500     # cắt doc compact (context 32K, 50×500 char OK)
    batch_size: int = 16             # cross-encoder pairs/batch
    use_instruction: bool = True     # thêm instruction cho Qwen3-Reranker (+1-5%)


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k: int = 10
    rerank_depth: int = 100
    engine: str = "qdrant"
    use_dense: bool = True           # Vector Index Search (HNSW) — kênh CHÍNH
    use_sparse: bool = True          # Qdrant sparse (TF/BM25 local) — kênh phụ
    fusion: str = "native"           # Qdrant FusionQuery(RRF) — dense + sparse trong 1 query
    statement_bonus: float = 0.05    # boost mềm khi table.statement == hint (KHÔNG hard filter)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    sparse: SparseConfig = Field(default_factory=SparseConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    embed_statement_only: bool = False  # knob: chỉ 10,797 bảng BCTC vs full 146K
    min_n_rows: int = 5              # bỏ bảng junk/TOC


class SandboxConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout: int = 20
    max_code_len: int = 8000
    max_ast_nodes: int = 800


class Config(BaseModel):
    """Cấu hình tổng — gộp base + api/local + env override."""

    model_config = ConfigDict(extra="forbid")

    paths: dict[str, Any] = Field(default_factory=dict)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    answer_abs_tol: float = 0.01
    step_budget: int = 10
    llm: LLMConfig = Field(default_factory=LLMConfig)
    # LLM phụ cho codegen (pandas expert) — nếu không có thì fallback sang llm.
    codegen_llm: LLMConfig | None = None

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
    def load(cls, config_path: Path | None = None, base_path: Path | None = None) -> "Config":
        """Load base.yaml rồi chồng config_path (vd api.yaml) lên.

        Extra ghi đè base theo từng key (deep merge cho llm/retrieval/sandbox) —
        base.yaml không được ném bỏ khi override thiếu field. `base_path` cho phép
        test truyền base tuỳ chỉnh; mặc định configs/base.yaml.
        """
        if config_path is None:
            config_path = ROOT / "configs" / "api.yaml"
        if base_path is None:
            base_path = ROOT / "configs" / "base.yaml"
        base = cls.from_yaml(base_path)
        extra = cls.from_yaml(config_path)
        merged = _deep_merge(base.model_dump(), extra.model_dump(exclude_unset=True))
        return cls.model_validate(merged)
