"""search.py — hybrid search trên Qdrant (dense HNSW + sparse, native RRF fusion).

Flow: entity filter → 1 query_points với prefetch [dense, sparse] + FusionQuery(RRF)
→ statement bonus mềm → top-N candidates (cho rerank) / top-k.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qdrant_client import QdrantClient, models

from vifinqa.retrieval.entity import Entities


@dataclass(frozen=True)
class SearchResult:
    report_id: str
    ticker: str
    year: int
    report_type: str
    table_id: str
    position: int
    page_no: int | None
    statement: str
    is_statement: bool
    unit: str
    unit_factor: float
    header_text: str
    row_labels: str
    score: float = 0.0
    dense_score: float | None = None
    sparse_score: float | None = None
    rank: int = field(default=0, compare=False)

    def relevant_tables_key(self) -> str:
        """Đơn vị nộp bài: `report_id|table_N` — khớp format chuẩn benchmark
        (DSKT-NOWJ/ViFinQA `common/schemas/table_ref.py`: `make_table_ref` →
        `f"{doc_name}|table_{table_id}"`, `parse_table_ref` tách bằng "|table_").
        `position` chính là N trong `table_N` (xem `etl/catalog_builder.py`:
        `table_id = f"table_{table_idx}"` → `parse_position`)."""
        return f"{self.report_id}|table_{self.position}"

    def compact_doc(self, max_chars: int = 500) -> str:
        """Text compact cho reranker — report_id|table_N, statement, header, labels."""
        stmt = self.statement or "thuyet minh"
        return (
            f"{self.report_id}|table_{self.position} | {stmt} | "
            f"{self.header_text} | {self.row_labels}"[:max_chars]
        )


def relevant_tables_key(r: SearchResult) -> str:
    return r.relevant_tables_key()


def build_payload_filter(entities: Entities) -> models.Filter | None:
    """Filter Qdrant từ entity (ticker IN, year IN, report_type match). None = global search."""
    must: list[models.FieldCondition] = []
    if entities.tickers:
        must.append(models.FieldCondition(key="ticker", match=models.MatchAny(any=sorted(entities.tickers))))
    if entities.years:
        must.append(models.FieldCondition(key="year", match=models.MatchAny(any=sorted(entities.years))))
    if entities.report_type:
        must.append(models.FieldCondition(key="report_type", match=models.MatchValue(value=entities.report_type)))
    if not must:
        return None
    return models.Filter(must=must)


def _payload_to_result(payload: dict, score: float) -> SearchResult:
    return SearchResult(
        report_id=payload.get("report_id", ""),
        ticker=payload.get("ticker", ""),
        year=int(payload.get("year", 0)),
        report_type=payload.get("report_type", ""),
        table_id=payload.get("table_id", ""),
        position=int(payload.get("position", 0)),
        page_no=payload.get("page_no"),
        statement=payload.get("statement") or "",
        is_statement=bool(payload.get("is_statement", False)),
        unit=payload.get("unit") or "",
        unit_factor=float(payload.get("unit_factor", 1.0)),
        header_text=payload.get("header_text") or "",
        row_labels=payload.get("row_labels") or "",
        score=score,
    )


def hybrid_search(
    client: QdrantClient,
    query_dense: list[float] | None,
    query_sparse: models.SparseVector | None,
    cfg,
    flt: models.Filter | None,
) -> list[SearchResult]:
    """Native hybrid query (dense + sparse prefetch + FusionQuery RRF) → SearchResult list."""
    col = cfg.qdrant.collection
    prefetch: list[models.Prefetch] = []
    if cfg.use_dense and query_dense is not None:
        prefetch.append(models.Prefetch(query=query_dense, using="dense", filter=flt, limit=cfg.rerank_depth))
    if cfg.use_sparse and query_sparse is not None:
        prefetch.append(models.Prefetch(query=query_sparse, using="sparse", filter=flt, limit=cfg.rerank_depth))
    if not prefetch:
        raise ValueError("use_dense và use_sparse đều False — không có channel nào.")

    if len(prefetch) == 1:
        resp = client.query_points(
            col, query=prefetch[0].query, using=prefetch[0].using,
            filter=flt, limit=cfg.rerank_depth, with_payload=True,
        )
    else:
        resp = client.query_points(
            col,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=cfg.rerank_depth,
            with_payload=True,
        )
    return [_payload_to_result(h.payload, float(h.score)) for h in resp.points]


def apply_statement_bonus(results: list[SearchResult], hint: str | None, bonus: float) -> list[SearchResult]:
    """Cộng bonus MỀM nếu table.statement == hint (notes table không bị phạt). Sort lại theo score."""
    if not hint or bonus == 0.0:
        return results
    out = []
    for i, r in enumerate(results):
        r = _replace_score(r, r.score + (bonus if r.statement == hint else 0.0))
        out.append(r)
    return sorted(out, key=lambda r: r.score, reverse=True)


def _replace_score(r: SearchResult, score: float) -> SearchResult:
    from dataclasses import replace

    return replace(r, score=score)
