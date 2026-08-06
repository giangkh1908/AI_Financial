"""run_retrieval.py — chạy retrieval trên N câu hỏi (mặc định 10) → topk CSV + metrics.

- Mặc định `--limit 10` — KHÔNG chạy full 1,012 lúc thử (tốn LLM rerank + thời gian).
- In metrics: coverage %, median/p95 latency, entity stats, facts-verification %, spot-check.

Chạy:  .venv\\Scripts\\python scripts/run_retrieval.py [--limit 10] [--spot-ids ...] [--no-rerank]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.config import Config  # noqa: E402
from vifinqa.retrieval.facts_index import FactsIndex  # noqa: E402
from vifinqa.retrieval.pipeline import RetrievalPipeline  # noqa: E402

DEFAULT_SPOT_IDS = "1,11,15,21,30,369,378,400,689,790"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="api.yaml")
    ap.add_argument("--limit", type=int, default=10, help="số câu chạy (0 = full 1,012)")
    ap.add_argument("--spot-ids", type=str, default=DEFAULT_SPOT_IDS, help="câu để in top-5 chi tiết")
    ap.add_argument("--out", type=str, default="data/derived/retrieval_topk.csv")
    ap.add_argument("--no-rerank", action="store_true", help="tắt bước rerank (test vector+sparse trần)")
    args = ap.parse_args()

    cfg = Config.load(ROOT / "configs" / args.config)
    if args.no_rerank:
        cfg.retrieval.rerank.enabled = False
    derived = cfg.resolved_derived_dir()

    questions_path = cfg.resolved_data_dir() / "questions" / "questions.jsonl"
    questions = []
    with open(questions_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            questions.append((int(obj["id"]), obj["question"]))

    spot = [int(x) for x in args.spot_ids.split(",") if x]
    if args.limit:
        # ưu tiên các câu spot-check rồi lấy thêm đầu danh sách cho đủ limit
        chosen = list(dict.fromkeys([i for i, _ in questions if i in spot]))
        for qid, _ in questions:
            if len(chosen) >= args.limit:
                break
            if qid not in chosen:
                chosen.append(qid)
        questions = [(i, q) for i, q in questions if i in chosen]

    print(f"Chạy {len(questions)} câu (rerank={not args.no_rerank})...")
    pipeline = RetrievalPipeline(cfg)
    facts = FactsIndex(derived / "facts_all.csv")

    t0 = time.time()
    lat = []
    rows = []
    spot_top = {}
    n_ticker, n_year = 0, 0
    n_cov = 0
    verify_acc = []
    try:
        for qid, question in questions:
            tq = time.time()
            results, entities = pipeline.search(question)
            dt = time.time() - tq
            lat.append(dt)
            if results:
                n_cov += 1
            if entities.tickers:
                n_ticker += 1
            if entities.years:
                n_year += 1
            if results:
                v = facts.verify(results)
                verify_acc.append(v["covered"])
            topk = ";".join(f"{r.relevant_tables_key()}|{r.score:.4f}" for r in results)
            rows.append({
                "id": qid, "question": question,
                "entity_json": json.dumps(entities.as_dict(), ensure_ascii=False),
                "topk": topk,
            })
            if qid in spot:
                spot_top[qid] = [(r.report_id, r.table_id, round(r.score, 4)) for r in results[:5]]
            print(f"  Q{qid}: {len(results)} kết quả ({dt:.1f}s) ticker={sorted(entities.tickers)}")
    finally:
        pipeline.close()

    # ghi CSV
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "question", "entity_json", "topk"])
        w.writeheader()
        w.writerows(rows)

    # metrics
    n = len(questions)
    lat_sorted = sorted(lat)
    median = lat_sorted[len(lat_sorted) // 2] if lat_sorted else 0
    p95 = lat_sorted[int(len(lat_sorted) * 0.95)] if len(lat_sorted) > 1 else (lat_sorted[-1] if lat_sorted else 0)
    metrics = {
        "n_questions": n,
        "coverage": n_cov / n if n else 0,
        "latency_median_s": round(median, 2),
        "latency_p95_s": round(p95, 2),
        "entity_ticker_rate": n_ticker / n if n else 0,
        "entity_year_rate": n_year / n if n else 0,
        "facts_covered_mean": (sum(verify_acc) / len(verify_acc)) if verify_acc else 0,
        "total_s": round(time.time() - t0, 1),
    }
    (derived / "retrieval_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== METRICS =====")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nTop-5 mỗi spot-check id {list(spot_top.keys())}:")
    for qid, tops in spot_top.items():
        print(f"  Q{qid}: {tops}")
    print(f"\nCSV: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
