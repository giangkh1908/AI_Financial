"""pipeline.py — orchestration retrieval: question → entity → hybrid search → rerank → top-k."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vifinqa.config import Config, ROOT
from vifinqa.retrieval.entity import load_company_map, extract_entities, Entities
from vifinqa.retrieval.index import make_embedder, make_qdrant_client, tf_sparse
from vifinqa.retrieval.search import (
    SearchResult,
    apply_statement_bonus,
    build_payload_filter,
    hybrid_search,
)


class RetrievalPipeline:
    """Pipeline truy hồi: lazy-load qdrant client + embed client + reranker (nếu bật)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        ret = cfg.retrieval
        self._cmap = load_company_map(cfg.resolved_data_dir() / "code_stock.csv")
        self._client = make_qdrant_client(ret)
        self._embedder = make_embedder(cfg)   # primary + fallbacks (local-first)
        self._reranker = None
        if ret.rerank.enabled:
            self._reranker = self._load_reranker(ret.rerank)

    def _load_reranker(self, rr):
        # Lazy import — tránh kéo torch/transformers khi rerank off (run_retrieval phục vụ không cần)
        from vifinqa.retrieval.rerank import LocalReranker

        model = rr.model
        path = Path(model)
        if not path.is_absolute() and (ROOT / path).exists():
            model = str(ROOT / path)
        return LocalReranker(model, device=rr.device)

    def _embed_query(self, question: str) -> np.ndarray:
        return self._embedder.embed([question])[0]

    def search(self, question: str) -> tuple[list[SearchResult], Entities]:
        """Trả (top-k SearchResult, Entities) — top-k theo thứ tự liên quan."""
        ret = self.cfg.retrieval
        entities = extract_entities(question, self._cmap)
        flt = build_payload_filter(entities)
        qdense = self._embed_query(question) if ret.use_dense else None
        qsparse = tf_sparse(question) if ret.use_sparse else None
        results = hybrid_search(self._client, qdense, qsparse, ret, flt)
        # Fallback: filter report_type quá hẹp (báo cáo 'other' không tách cons/sep —
        # EVF/FTS...). "công ty mẹ"→separate nhưng chỉ có 1 báo cáo → bỏ filter type.
        if not results and flt is not None and entities.report_type:
            relaxed = build_payload_filter(entities.__class__(
                tickers=entities.tickers, years=entities.years,
                year_ranges=entities.year_ranges, report_type=None,
                statement=entities.statement, unit_factor=entities.unit_factor,
                unit_label=entities.unit_label, matched_names=entities.matched_names,
            ))
            results = hybrid_search(self._client, qdense, qsparse, ret, relaxed)
        results = apply_statement_bonus(results, entities.statement, ret.statement_bonus)
        results = results[: ret.rerank.candidates]
        if ret.rerank.enabled and results and self._reranker is not None:
            results = self._rerank(question, results, ret.rerank)
        return results[: ret.k], entities

    def _rerank(self, question: str, results: list[SearchResult], rr) -> list[SearchResult]:
        pairs = [(question, r.compact_doc(rr.max_chars_per_doc)) for r in results]
        scores = self._reranker.score_pairs(pairs, batch_size=rr.batch_size)
        ranked = sorted(zip(results, scores), key=lambda rs: rs[1], reverse=True)
        return [r for r, _ in ranked]

    def close(self) -> None:
        self._client.close()
