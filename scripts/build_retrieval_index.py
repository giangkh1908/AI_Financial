"""build_retrieval_index.py — build index retrieval (M3) toàn corpus vào Qdrant local.

- Đọc `catalog_tables.csv` (146K bảng, multi-line) một lần, nhóm theo ticker.
- Per ticker: chunk `text_dense`/`text_lex` → embed dense (baai/bge-m3, OpenRouter API,
  cache `.npy` per ticker → resume không trả phí lại) + sparse TF local → upsert Qdrant
  (dense HNSW INT8 + sparse `modifier: idf`).
- Checkpoint `data/derived/retrieval_state.json` {ticker:1}; lỗi → `retrieval_errors.tsv`.
- Qdrant local = single process (file lock path) → chạy tuần tự.

Chạy:  .venv\\Scripts\\python scripts/build_retrieval_index.py [--config api.yaml] [--tickers HPG,VCB]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Console Windows (cp1252) không in được tiếng Việt → ép UTF-8
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.config import Config  # noqa: E402
from vifinqa.retrieval.index import (  # noqa: E402
    _new_openai_client,
    build_ticker,
    embed_dense,
    ensure_collection,
    iter_catalog_tables,
    load_fact_labels,
    make_qdrant_client,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1, help="giữ tương thích; qdrant local là single process")
    ap.add_argument("--config", type=str, default="api.yaml")
    ap.add_argument("--tickers", type=str, default="", help="chỉ chạy các ticker, phân cách ','")
    args = ap.parse_args()

    cfg = Config.load(ROOT / "configs" / args.config)
    ret = cfg.retrieval
    derived_dir = cfg.resolved_derived_dir()
    cache_dir = Path(ret.embedding.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir  # cache_dir đã là path tính từ ROOT (vd "data/derived/embeddings")
    cache_dir.mkdir(parents=True, exist_ok=True)

    client = make_qdrant_client(ret)

    try:
        catalog_path = derived_dir / "catalog_tables.csv"
        facts_path = derived_dir / "facts_all.csv"
        ensure_collection(client, ret)
        print(f"Collection '{ret.qdrant.collection}' sẵn sàng (HNSW m={ret.qdrant.hnsw_m}, ef_construct={ret.qdrant.hnsw_ef_construct}, INT8={ret.qdrant.quantize}, sparse={ret.qdrant.sparse_modifier})")

        # nhóm catalog rows theo ticker (đọc 1 lần)
        t0 = time.time()
        by_ticker: dict[str, list[dict]] = defaultdict(list)
        total_rows = 0
        for row in iter_catalog_tables(catalog_path, ret):
            by_ticker[row["ticker"]].append(row)
            total_rows += 1
        print(f"Catalog: {total_rows} bảng, {len(by_ticker)} ticker ({time.time()-t0:.0f}s)")

        fact_labels = load_fact_labels(facts_path)
        print(f"Facts labels: {len(fact_labels)} key (ticker,year,type,statement)")

        state_path = derived_dir / "retrieval_state.json"
        state: dict[str, int] = {}
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
        errors_path = derived_dir / "retrieval_errors.tsv"

        wanted = set(args.tickers.split(",")) if args.tickers else None
        todo = [t for t in by_ticker if (not wanted or t in wanted) and t not in state]
        print(f"Ticker tổng: {len(by_ticker)}, đã xong: {len(by_ticker) - len(todo)}, sẽ chạy: {len(todo)}")

        embed_client = None
        if ret.embedding.provider == "openrouter":
            embed_client = _new_openai_client(cfg)  # chỉ cần client OpenAI cho openrouter
        total_pts = 0
        total_chars = 0
        total_hits = 0
        t0 = time.time()
        for i, ticker in enumerate(todo, 1):
            try:
                rows = by_ticker[ticker]
                n_pts, hits = build_ticker(client, embed_client, rows, ret, fact_labels, cache_dir)
                total_pts += n_pts
                total_hits += hits
                total_chars += sum(len(r["header_text"]) + len(r["row_labels"]) + len(r["anchor_context"]) for r in rows)
                state[ticker] = 1
                state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                with open(errors_path, "a", encoding="utf-8", newline="") as f:
                    f.write(f"{ticker}\t{type(e).__name__}\t{e}\n")
                print(f"  [err] {ticker}: {e}", file=sys.stderr)
            if i % 20 == 0 or i == len(todo):
                print(f"  ...{i}/{len(todo)} ticker, {time.time()-t0:.0f}s")

        n_pts = client.count(ret.qdrant.collection, exact=True).count
        est_tokens = total_chars / 4.0
        est_cost = est_tokens * 0.01 / 1e6
        print(f"\nHoàn tất {len(todo)} ticker trong {time.time()-t0:.0f}s.")
        print(f"Points trong collection: {n_pts} (catalog rows: {total_rows})")
        print(f"Cache hit ticker: {total_hits}/{len(todo)}")
        print(f"Ước lượng chi phí embed (bge-m3 $0.01/1M): ~${est_cost:.2f} cho ~{est_tokens/1e6:.1f}M token")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
