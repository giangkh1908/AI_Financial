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
import re
import uuid
from pathlib import Path

import numpy as np
from openai import OpenAI
from qdrant_client import QdrantClient, models

from vifinqa.config import Config, RetrievalConfig, ROOT
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
    }


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


def _new_openai_client(cfg: Config) -> OpenAI:
    llm = cfg.llm
    return OpenAI(
        base_url=llm.base_url,
        api_key=llm.effective_api_key(),
        timeout=llm.timeout,
        max_retries=llm.retries,
        default_headers=llm.extra_headers or None,
    )


def embed_dense(client: OpenAI, texts: list[str], cfg: RetrievalConfig) -> np.ndarray:
    """Embed batch qua OpenRouter Embeddings API → [N, dense_dim] float32.

    Các API calls độc lập → chạy song song (`embedding.workers`) để full corpus
    ~10-15 phút thay vì ~1h tuần tự. OpenAI client thread-safe.
    """
    from concurrent.futures import ThreadPoolExecutor

    batch = cfg.embedding.batch_size
    batches = [texts[i : i + batch] for i in range(0, len(texts), batch)]
    results: list[list[list[float]]] = [None] * len(batches)  # type: ignore[list-item]

    def _run(i: int, chunk: list[str]) -> tuple[int, list[list[float]]]:
        import random as _random
        import time as _time
        from openai import APIStatusError, APITimeoutError, APIConnectionError

        for attempt in range(5):  # retry 429/5xx/timeout với backoff mũ + jitter
            try:
                resp = client.embeddings.create(model=cfg.embedding.model, input=chunk)
                return i, [d.embedding for d in resp.data]
            except (APITimeoutError, APIConnectionError) as e:
                if attempt == 4:
                    raise
                _time.sleep((2 ** attempt) + _random.uniform(0, 1))
            except APIStatusError as e:
                # Chỉ retry lỗi tạm thời (429 overloaded, 5xx). 401/403/404 raise ngay.
                if attempt == 4 or e.status_code not in (429, 500, 502, 503, 504):
                    raise
                _time.sleep((2 ** attempt) + _random.uniform(0, 1))

    workers = min(cfg.embedding.workers, len(batches) or 1)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, vecs in ex.map(_run, range(len(batches)), batches):
            results[i] = vecs
    flat = [v for chunk in results for v in chunk]
    return np.asarray(flat, dtype=np.float32)


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


def build_ticker(
    client: QdrantClient,
    embed_client: OpenAI,
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
        dense = embed_dense(embed_client, texts_dense, cfg)
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
