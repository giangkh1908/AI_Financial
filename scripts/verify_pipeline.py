"""verify_pipeline.py — verify 30 câu: retrieval + deterministic + giá trị tidy.

Không gọi LLM. In: entity, top-1 bảng, tier match, value trả về, evidence path tồn tại.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.config import Config
from vifinqa.agent.loop import _plan_evidence
from vifinqa.engine.deterministic import solve_deterministic, is_complex
from vifinqa.retrieval.facts_index import FactsIndex
from vifinqa.retrieval.pipeline import RetrievalPipeline

IDS = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 25, 26, 27, 30, 31, 32, 33, 35, 36]


def main() -> int:
    cfg = Config.load(ROOT / "configs" / "api.yaml")
    cfg.retrieval.rerank.enabled = False  # verify nhanh không cần rerank
    derived = cfg.resolved_derived_dir()

    qs = {}
    with open(cfg.resolved_data_dir() / "questions" / "questions.jsonl", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if int(o["id"]) in IDS:
                qs[int(o["id"])] = o["question"]

    pipeline = RetrievalPipeline(cfg)
    facts = FactsIndex(derived / "facts_all.csv")

    print(f"=== Verify {len(qs)} câu (deterministic, no-LLM, no-rerank) ===")
    n_ok = 0
    n_tier = {"facts": 0, "evidence": 0, "none": 0}
    try:
        for qid in IDS:
            q = qs[qid]
            t0 = time.time()
            results, entities = pipeline.search(q)
            results = results or []
            usable, cards, evidence, plan = _plan_evidence(results, facts, derived)
            complex_q = is_complex(q)
            det = None
            if not complex_q and len(entities.tickers) == 1 and entities.years:
                det = solve_deterministic(q, entities, facts, list(evidence.values()), plan)
            tier = "facts" if det and det["tier"] == "facts" else ("evidence" if det else "none")
            n_tier[tier] += 1
            if det:
                n_ok += 1
            top1 = f"{usable[0].report_id}|{usable[0].table_id}({usable[0].statement or 'notes'})" if usable else "KHÔNG CÓ"
            ans = det["answer"] if det else None
            ev_names = [p["name"] for p in plan][:4]
            print(f"  Q{qid} [{tier:>8}] ans={ans} | top1={top1} | ev={ev_names} | {time.time()-t0:.1f}s")
    finally:
        pipeline.close()

    print(f"\n=== Kết quả: {n_ok}/{len(qs)} deterministic match | tier {n_tier} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
