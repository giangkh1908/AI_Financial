"""index.py — build offline: chunking + dense embed (OpenRouter API + cache) + sparse TF + upsert Qdrant.

Mỗi bảng → 2 biểu diễn:
- `text_dense`: prefix metadata + header_text + row_labels + anchor (+ item_label sạch từ facts
  nếu là BCTC) → embed dense (baai/bge-m3, 1024-dim) qua OpenRouter API, cache `.npy` per ticker.
- `text_lex`: cùng nội dung đã bỏ dấu → TF sparse (thuần Python) → Qdrant sparse vector
  (`modifier: idf` tự tính IDF).

Point id = uuid5(report_id|table_id) → idempotent (resume = re-upsert không trùng).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from openai import OpenAI
from qdrant_client import QdrantClient, models

from vifinqa.config import Config, EmbeddingConfig, EmbedProviderRef, RetrievalConfig, ROOT
from vifinqa.etl.numbers import normalize_label

# Namespace cố định cho point id (không đổi giữa các lần build)
_UUID_NS = uuid.UUID("3e2a1f0c-9b4d-4f2e-8c1a-5b7d6e9f0a1b")
# Không gian term-id cho sparse: 2^21 bucket, hash md5 (ổn định giữa process — không dùng hash() Python)
_SPARSE_BITS = 21


def point_id(report_id: str, table_id: str) -> str:
    return str(uuid.uuid5(_UUID_NS, f"{report_id}|{table_id}"))


def make_qdrant_client(cfg: RetrievalConfig) -> QdrantClient:
    """Factory QdrantClient theo cfg.qdrant.mode — server (Docker localhost) hoặc local (embedded path).

    Dùng chung cho build + serve → tránh trùng logic path-resolution. Path local resolve
    tương tự qdrant.path (ROOT-anchored khi relative).
    """
    q = cfg.qdrant
    if q.mode == "local":
        qp = Path(q.path)
        if not qp.is_absolute():
            qp = ROOT / qp
        return QdrantClient(path=str(qp))
    return QdrantClient(host=q.host, port=q.port)


def parse_position(table_id: str) -> int:
    return int(table_id.removeprefix("table_"))


def _compact(s: str) -> str:
    """normalize_label + thay ký tự không phải [a-z0-9] bằng space."""
    return re.sub(r"[^a-z0-9]+", " ", normalize_label(s)).strip()


def tokenize(text: str) -> list[str]:
    return _compact(text).split()


def _term_id(term: str) -> int:
    return int(hashlib.md5(term.encode("utf-8")).hexdigest()[:8], 16) % (1 << _SPARSE_BITS)


def tf_sparse(text: str) -> models.SparseVector:
    """TF weights → Qdrant SparseVector (indices tăng dần, dedupe hash-collision bằng cộng trọng số)."""
    counts: dict[int, float] = {}
    for term in tokenize(text):
        tid = _term_id(term)
        counts[tid] = counts.get(tid, 0.0) + 1.0
    idx = sorted(counts)
    return models.SparseVector(indices=idx, values=[counts[i] for i in idx])


def _truncate_chars(s: str, max_chars: int) -> str:
    return s if len(s) <= max_chars else s[:max_chars]


def build_table_chunks(row: dict, cfg: RetrievalConfig, fact_labels: list[str] | None = None) -> tuple[str, str]:
    """(text_dense, text_lex) cho 1 bảng — chuẩn theo cấu trúc catalog + facts."""
    stmt = row["statement"] or "BẢNG THUYẾT MINH"
    prefix = f"BÁO CÁO {row['report_type']} {row['year']} | {stmt} | {row['ticker']}"
    header = (row["header_text"] or "").strip()
    labels = (row["row_labels"] or "").strip()
    anchor = (row["anchor_context"] or "").strip()

    def _parts() -> list[str]:
        parts = [p for p in (prefix, header, labels, anchor) if p]
        if fact_labels:
            parts.append(" | ".join(dict.fromkeys(fact_labels)))  # dedupe giữ thứ tự
        return parts

    text_dense = _truncate_chars("\n".join(_parts()), cfg.embedding.max_chars)
    text_lex = _truncate_chars(_compact("\n".join(_parts())), cfg.sparse.max_chars)
    return text_dense, text_lex


def build_payload(row: dict, text_dense: str) -> dict:
    page_no = (row["page_no"] or "").strip()
    return {
        "report_id": row["report_id"],
        "ticker": row["ticker"],
        "year": int(row["year"]),
        "report_type": row["report_type"],
        "table_id": row["table_id"],
        "position": parse_position(row["table_id"]),
        "page_no": int(page_no) if page_no.isdigit() else None,
        "statement": row["statement"],
        "is_statement": row["is_statement"] == "1",
        "unit": row["unit"],
        "unit_factor": float(row["unit_factor"]) if row["unit_factor"] else 1.0,
        "header_text": _truncate_chars(row["header_text"] or "", 400),
        "row_labels": _truncate_chars(row["row_labels"] or "", 600),
        "period_cols": _period_cols_list(row.get("period_cols")),
    }


def _period_cols_list(s) -> list[int]:
    """'[3, 4]' (JSON trong catalog) → [3, 4]; fallback parse '3;4'."""
    if not s:
        return []
    import json as _json

    try:
        v = _json.loads(s)
        return [int(x) for x in v] if isinstance(v, list) else []
    except (ValueError, TypeError):
        pass
    out = []
    for part in str(s).replace("[", "").replace("]", "").split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out


def iter_catalog_tables(catalog_path: Path, cfg: RetrievalConfig, tickers: set[str] | None = None):
    """Đọc catalog_tables.csv (multi-line rows — bắt buộc csv.DictReader) theo filter."""
    with open(catalog_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if tickers and row["ticker"] not in tickers:
                continue
            if cfg.embed_statement_only and row["is_statement"] != "1":
                continue
            try:
                n_rows = int(row["n_rows"])
            except (TypeError, ValueError):
                n_rows = 0
            if n_rows < cfg.min_n_rows:
                continue
            yield row


def load_fact_labels(facts_all_path: Path) -> dict[tuple[str, int, str, str], list[str]]:
    """{(ticker, year, report_type, statement): [item_label dedupe]} — cho enrichment text_dense."""
    import pandas as pd

    df = pd.read_csv(facts_all_path, dtype={"year": int}, low_memory=False)
    out: dict[tuple[str, int, str, str], list[str]] = {}
    for (tk, yr, rt, st), grp in df.groupby(["ticker", "year", "report_type", "statement"]):
        labels = grp["item_label"].dropna().astype(str).tolist()
        out[(tk, int(yr), str(rt), str(st))] = list(dict.fromkeys(labels))
    return out


@dataclass
class _EmbedEndpoint:
    """Một endpoint embed trong fallback chain.

    kind='openai'   → OpenAI-compatible embeddings API (deepinfra/openai_compatible/openrouter/vllm).
    kind='http_bge' → HTTP `/embed` của bge_m3_server (provider ngrok/http_bge, không cần SDK/key).
    """
    kind: str
    base_url: str
    model: str
    dim: int
    client: OpenAI | None = None      # cho kind='openai'
    provider: str = ""


_HTTP_PROVIDERS = {"ngrok", "http_bge"}


def _endpoint_from_embedding(emb: EmbeddingConfig) -> _EmbedEndpoint:
    """Primary endpoint từ cfg.retrieval.embedding (EmbeddingConfig)."""
    if emb.provider in _HTTP_PROVIDERS:
        return _EmbedEndpoint("http_bge", emb.base_url, emb.model, emb.dense_dim, None, emb.provider)
    client = OpenAI(base_url=emb.base_url, api_key=emb.effective_api_key(), timeout=60.0, max_retries=1)
    return _EmbedEndpoint("openai", emb.base_url, emb.model, emb.dense_dim, client, emb.provider)


def _endpoint_from_ref(ref: EmbedProviderRef) -> _EmbedEndpoint:
    """Fallback endpoint từ cfg.retrieval.embedding.fallbacks[i] (EmbedProviderRef)."""
    if ref.provider in _HTTP_PROVIDERS:
        return _EmbedEndpoint("http_bge", ref.base_url, ref.model, ref.dense_dim, None, ref.provider)
    client = OpenAI(base_url=ref.base_url, api_key=ref.effective_api_key(), timeout=60.0, max_retries=1)
    return _EmbedEndpoint("openai", ref.base_url, ref.model, ref.dense_dim, client, ref.provider)


class Embedder:
    """Fallback chain embed: primary + fallbacks, sticky-first + cooldown 30s.

    Thread-safe (sticky/cooldown dưới lock; bản thân call embed I/O chạy song song).
    Dim guard: mọi endpoint phải cùng dim với primary — tránh embed sai dim làm hỏng index.
    """

    _COOLDOWN = 30.0

    def __init__(self, endpoints: list[_EmbedEndpoint]):
        if not endpoints:
            raise ValueError("Embedder cần ≥1 endpoint")
        self._eps = endpoints
        self._dim = endpoints[0].dim
        for ep in endpoints:
            if ep.dim != self._dim:
                raise ValueError(
                    f"Embed fallback dim mismatch: endpoint {ep.provider} {ep.base_url} "
                    f"dim={ep.dim} ≠ primary dim={self._dim}. Qdrant collection fix dim → "
                    f"phải cùng dim tất cả fallback."
                )
        self._sticky = 0
        self._failed_until: dict[int, float] = {}
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        return self._dim

    def _order(self) -> list[int]:
        now = time.time()
        alive = [i for i in range(len(self._eps)) if self._failed_until.get(i, 0.0) <= now]
        if not alive:
            alive = list(range(len(self._eps)))
        with self._lock:
            sticky = self._sticky
        if sticky in alive:
            return [sticky] + [i for i in alive if i != sticky]
        return alive

    def _embed_batch(self, idx: int, chunk: list[str]) -> list[list[float]]:
        """Embed 1 batch qua endpoint idx (in-provider retry 5 lần transient)."""
        ep = self._eps[idx]
        if ep.kind == "http_bge":
            return _ngrok_embed(ep.base_url, chunk)
        assert ep.client is not None
        resp = ep.client.embeddings.create(model=ep.model, input=chunk, encoding_format="float")
        return [d.embedding for d in resp.data]

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed list text → [N, dim] float32. Thử chain; transient endpoint fail → nhảy.

        KHÔNG tự retry ở cấp chain cho 1 batch (embed_dense đã chạy batch song song và
        từng batch có retry-in-provider trong _run). Endpoint fail toàn bộ → đánh dấu
        cooldown rồi raise để embed_dense bắt và thử lại batch ở endpoint khác.
        """
        idx = self._order()[0]  # batch này dùng endpoint sticky/alive đầu tiên
        try:
            vecs = self._embed_batch(idx, texts)
            with self._lock:
                self._sticky = idx
            return np.asarray(vecs, dtype=np.float32)
        except Exception:
            with self._lock:
                self._failed_until[idx] = time.time() + self._COOLDOWN
            # thử endpoint kế tiếp nếu còn
            for idx2 in self._order():
                if idx2 == idx:
                    continue
                try:
                    vecs = self._embed_batch(idx2, texts)
                    with self._lock:
                        self._sticky = idx2
                    return np.asarray(vecs, dtype=np.float32)
                except Exception:
                    with self._lock:
                        self._failed_until[idx2] = time.time() + self._COOLDOWN
            raise  # cả chain fail → raise (embed_dense/batch log error)


def make_embedder(cfg: Config) -> Embedder:
    """Build Embedder từ cfg.retrieval.embedding (primary) + .fallbacks."""
    emb = cfg.retrieval.embedding
    eps = [_endpoint_from_embedding(emb)]
    for ref in emb.fallbacks:
        eps.append(_endpoint_from_ref(ref))
    return Embedder(eps)


def _ngrok_embed(url: str, texts: list[str], timeout: float = 120.0) -> list[list[float]]:
    """Gọi `/embed` (BGE-M3 FastAPI trên GPU thuê) → list dense vectors (JSON)."""
    import json as _json
    import urllib.error as _urlerr
    import urllib.request as _urlreq

    body = _json.dumps({"texts": texts, "return_dense": True}).encode("utf-8")
    req = _urlreq.Request(url.rstrip("/") + "/embed", data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with _urlreq.urlopen(req, timeout=timeout) as r:
            resp = _json.loads(r.read().decode("utf-8"))
    except _urlerr.HTTPError as e:
        raise ConnectionError(f"ngrok /embed HTTP {e.code}: {e.read().decode()[:200]}") from e
    dense = resp.get("dense")
    if not dense:
        raise ValueError(f"ngrok /embed không trả dense: {resp}")
    return dense


def embed_dense(embedder: Embedder, texts: list[str], cfg: RetrievalConfig) -> np.ndarray:
    """Embed batch → [N, dense_dim] float32 qua Embedder (chain fallback).

    Chia nhỏ theo embedding.batch_size, chạy song song (embedding.workers). Mỗi batch tự
    qua Embedder.embed (sticky-first + cooldown); cả chain fail → retry transient 5 lần rồi
    raise (build script log error per ticker, giữ batch đi tiếp).
    """
    from concurrent.futures import ThreadPoolExecutor

    emb = cfg.embedding
    if not texts:
        return np.zeros((0, embedder.dim), dtype=np.float32)
    batch = emb.batch_size
    batches = [texts[i : i + batch] for i in range(0, len(texts), batch)]
    results: list[np.ndarray] = [None] * len(batches)  # type: ignore[list-item]

    def _run(i: int, chunk: list[str]) -> tuple[int, np.ndarray]:
        import random as _random

        for attempt in range(5):  # retry transient lỗi với backoff mũ + jitter
            try:
                return i, embedder.embed(chunk)
            except Exception as e:  # noqa: BLE001 — retry tạm thời (timeout/5xx/conn)
                transient = any(t in type(e).__name__ for t in ("Timeout", "Connection", "HTTP")) or "429" in str(e)
                if attempt == 4 or not transient:
                    raise
                time.sleep((2 ** attempt) + _random.uniform(0, 1))

    workers = min(emb.workers, len(batches) or 1)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, vecs in ex.map(_run, range(len(batches)), batches):
            results[i] = vecs
    arrs = [r for r in results if r is not None]
    return np.concatenate(arrs, axis=0)


def _texts_hash(texts: list[str]) -> str:
    return hashlib.md5("\x00".join(texts).encode("utf-8")).hexdigest()


def ensure_collection(client: QdrantClient, cfg: RetrievalConfig) -> None:
    col = cfg.qdrant.collection
    if client.collection_exists(col):
        return
    quant = None
    if cfg.qdrant.quantize:
        quant = models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True, quantile=0.99)
        )
    client.create_collection(
        collection_name=col,
        vectors_config={
            "dense": models.VectorParams(
                size=cfg.embedding.dense_dim,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(
                    m=cfg.qdrant.hnsw_m,
                    ef_construct=cfg.qdrant.hnsw_ef_construct,
                    on_disk=cfg.qdrant.hnsw_on_disk,
                ),
                quantization_config=quant,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=cfg.qdrant.sparse_on_disk),
                modifier=models.Modifier.IDF if cfg.qdrant.sparse_modifier == "idf" else models.Modifier.NONE,
            )
        },
    )
    # Payload index tường minh — filter trong vector search (HNSW) cần index; nếu để
    # Qdrant tự tạo, có thể trả 0 kết quả khi kết hợp nhiều field (AND) giữa lúc build
    # (đã gặp: ticker+year filter → 0 dù payload có, scroll lại OK).
    for field_name, field_schema in (
        ("year", models.PayloadSchemaType.INTEGER),
        ("ticker", models.PayloadSchemaType.KEYWORD),
        ("report_type", models.PayloadSchemaType.KEYWORD),
        ("statement", models.PayloadSchemaType.KEYWORD),
    ):
        try:
            client.create_payload_index(col, field_name=field_name, field_schema=field_schema)
        except Exception:  # noqa: BLE001 — đã tồn tại / không hỗ trợ → bỏ qua
            pass


def build_ticker(
    client: QdrantClient,
    embedder: Embedder,
    rows: list[dict],
    cfg: RetrievalConfig,
    fact_labels: dict[tuple[str, int, str, str], list[str]],
    cache_dir: Path,
) -> tuple[int, int]:
    """Build 1 ticker → upsert Qdrant. Trả (n_points, n_cache_hit).

    Flow: (1) build chunks+payloads, (2) embed dense (cache per ticker), (3) build points
    với cả dense+sparse đã fill, (4) upsert batch 256. Không dùng placeholder zeros.
    """
    if not rows:
        return 0, 0

    # (1) chunks + payloads
    texts_dense: list[str] = []
    texts_lex: list[str] = []
    payloads: list[dict] = []
    for row in rows:
        key = (row["ticker"], int(row["year"]), row["report_type"], row["statement"])
        td, tl = build_table_chunks(row, cfg, fact_labels.get(key))
        texts_dense.append(td)
        texts_lex.append(tl)
        payloads.append(build_payload(row, td))

    # (2) embed dense — cache .npy per ticker; hit key = hash(texts)+n+model+dim
    ticker = rows[0]["ticker"]
    npy_path = cache_dir / f"{ticker}.npy"
    side_path = cache_dir / f"{ticker}.json"
    h = _texts_hash(texts_dense)
    cache_hit = False
    if npy_path.exists() and side_path.exists():
        try:
            side = json.loads(side_path.read_text(encoding="utf-8"))
            if (
                side.get("hash") == h
                and side.get("n") == len(texts_dense)
                and side.get("model") == cfg.embedding.model
                and side.get("dim") == cfg.embedding.dense_dim
            ):
                dense = np.load(npy_path)
                cache_hit = (
                    dense.shape[0] == len(texts_dense)
                    and dense.shape[1] == cfg.embedding.dense_dim
                )
        except Exception:
            cache_hit = False
    if not cache_hit:
        dense = embed_dense(embedder, texts_dense, cfg)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, dense)
        side_path.write_text(
            json.dumps({
                "hash": h, "n": len(texts_dense),
                "model": cfg.embedding.model, "dim": cfg.embedding.dense_dim,
            }),
            encoding="utf-8",
        )

    # (3)+(4) build points (dense list + sparse) rồi upsert batch
    points = [
        models.PointStruct(
            id=point_id(row["report_id"], row["table_id"]),
            payload=payload,
            vector={"dense": vec.tolist(), "sparse": tf_sparse(tl)},
        )
        for row, payload, vec, tl in zip(rows, payloads, dense, texts_lex)
    ]
    for i in range(0, len(points), 256):
        client.upsert(cfg.qdrant.collection, points=points[i : i + 256])
    return len(points), (1 if cache_hit else 0)
