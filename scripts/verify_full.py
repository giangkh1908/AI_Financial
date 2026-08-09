"""verify_full.py — test index+lexical trên TOÀN BỘ 1012 câu (no-LLM, no-rerank).

Mục đích: đo index hoạt động thế nào ở quy mô full:
- % câu có bảng truy hồi (coverage)
- % deterministic match (facts/evidence)
- trung bình số bảng evidence / số bảng lexical thêm
- phân bố tier

Chạy:  python scripts/verify_full.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.agent.loop import _label_recall, _plan_evidence
from vifinqa.config import Config
from vifinqa.engine.deterministic import is_complex, solve_deterministic
from vifinqa.loader import load_questions
from vifinqa.retrieval.facts_index import FactsIndex
from vifinqa.retrieval.pipeline import RetrievalPipeline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = full 1012")
    ap.add_argument("--workers", type=int, default=1, help="threads (embed query tốn API, giữ thấp)")
    args = ap.parse_args()

    cfg = Config.load(ROOT / "configs" / "api.yaml")
    cfg.retrieval.rerank.enabled = False
    derived = cfg.resolved_derived_dir()

    questions = load_questions(cfg.resolved_data_dir() / "questions" / "questions.jsonl")
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    pipeline = RetrievalPipeline(cfg)
    facts = FactsIndex(derived / "facts_all.csv")

    print(f"=== Verify index trên {len(questions)} câu (no-LLM) ===")
    n_cov = 0          # có ≥1 bảng truy hồi
    n_lex_add = 0      # câu có lexical thêm bảng
    total_lex = 0
    n_tier = {"facts": 0, "evidence": 0, "none": 0}
    n_ev_avg = 0
    t0 = time.time()

    from concurrent.futures import ThreadPoolExecutor

    def _one(q):
        qid = q["id"]
        question = q["question"]
        results, entities = pipeline.search(question)
        results = results or []
        n_pre = len(results)
        results = _label_recall(question, entities, results, derived)
        n_post = len(results)
        usable, cards, evidence, plan = _plan_evidence(results, facts, derived)
        complex_q = is_complex(question)
        det = None
        if not complex_q and len(entities.tickers) == 1 and entities.years:
            det = solve_deterministic(question, entities, facts, list(evidence.values()), plan)
        tier = "facts" if det and det["tier"] == "facts" else ("evidence" if det else "none")
        return {
            "qid": qid, "n_pre": n_pre, "n_post": n_post,
            "n_ev": len(evidence), "tier": tier, "n_lex_add": max(0, n_post - n_pre),
            "top1": f"{usable[0].report_id}|{usable[0].table_id}" if usable else "NONE",
            "entities": f"{sorted(entities.tickers)}_{sorted(entities.years)}_{entities.report_type}",
        }

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_one, questions), 1):
            if r["n_pre"] > 0:
                n_cov += 1
            if r["n_lex_add"] > 0:
                n_lex_add += 1
                total_lex += r["n_lex_add"]
            n_tier[r["tier"]] += 1
            n_ev_avg += r["n_ev"]
            if i % 100 == 0 or i == len(questions):
                print(f"  ...{i}/{len(questions)} | cov={n_cov} det={n_tier['facts']+n_tier['evidence']} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    pipeline.close()
    n = len(questions)
    print(f"\n=== KẾT QUẢ ({n} câu, {time.time()-t0:.0f}s) ===")
    print(f"Coverage (có ≥1 bảng truy hồi): {n_cov} ({n_cov/n*100:.1f}%)")
    print(f"Deterministic match: {n_tier['facts']+n_tier['evidence']} "
          f"(facts={n_tier['facts']}, evidence={n_tier['evidence']}) ({ (n_tier['facts']+n_tier['evidence'])/n*100:.1f}%)")
    print(f"  → còn {n_tier['none']} ({n_tier['none']/n*100:.1f}%) cần LLM/fallback")
    print(f"Lexical recall thêm bảng: {n_lex_add} câu ({n_lex_add/n*100:.1f}%), tổng {total_lex} bảng")
    print(f"Trung bình evidence/bảng: {n_ev_avg/n:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
