"""run_codegen_ids.py — chạy codegen cho danh sách id cụ thể (test câu khó)."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.agent import solve  # noqa: E402
from vifinqa.codegen import LLMClient  # noqa: E402
from vifinqa.config import Config  # noqa: E402
from vifinqa.loader import load_questions  # noqa: E402
from vifinqa.retrieval.facts_index import FactsIndex  # noqa: E402
from vifinqa.retrieval.pipeline import RetrievalPipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="JSON file chứa list id hoặc '450,460,...'")
    ap.add_argument("--config", default=str(ROOT / "configs" / "api.yaml"))
    ap.add_argument("--out", default=str(ROOT / "data" / "out" / "results_hard20.jsonl"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    ids_file = Path(args.ids)
    if ids_file.exists():
        ids = set(json.loads(ids_file.read_text(encoding="utf-8")))
    else:
        ids = {int(x) for x in args.ids.split(",") if x.strip()}

    cfg = Config.load(Path(args.config))
    questions = [q for q in load_questions(cfg.resolved_data_dir() / "questions" / "questions.jsonl") if q["id"] in ids]
    questions.sort(key=lambda q: q["id"])
    print(f"Chạy {len(questions)} câu khó (ids: {sorted(ids)[:5]}...{sorted(ids)[-3:]})", flush=True)

    pipeline = RetrievalPipeline(cfg)
    facts_index = FactsIndex(cfg.resolved_derived_dir() / "facts_all.csv")
    llm = LLMClient(cfg)  # head LLM (Qwen)
    codegen_llm = LLMClient(cfg, section="codegen_llm") if cfg.codegen_llm else None  # deepseek-coder

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_fail = 0

    # WALL_CAP: câu nào không xong trong X giây → fallback, không kẹt cả batch
    WALL_CAP = float(cfg.sandbox.timeout) * 6.0  # 20s pandas ×6 = 120s/question

    def _do(q):
        qres: queue.Queue = queue.Queue()

        def _worker():
            try:
                qres.put(solve(q["question"], q["id"], pipeline, facts_index, llm, cfg, codegen_llm=codegen_llm))
            except Exception as e:  # guard lỗi
                qres.put({"_exc": e})

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        try:
            return qres.get(timeout=WALL_CAP)
        except queue.Empty:
            return None

    with open(out_path, "w", encoding="utf-8") as fout:
        for i, q in enumerate(questions, 1):
            rec = _do(q)
            if rec is None:
                rec = {
                    "id": q["id"], "question": q["question"], "answer": 0.0,
                    "relevant_docs": [], "relevant_tables": [], "evidence": [],
                    "pandas_query": "result = 0.0", "_ok": False,
                    "_error": f"wall-cap timeout {WALL_CAP:.0f}s",
                }
            elif isinstance(rec, dict) and "_exc" in rec:
                rec = {
                    "id": q["id"], "question": q["question"], "answer": 0.0,
                    "relevant_docs": [], "relevant_tables": [], "evidence": [],
                    "pandas_query": "result = 0.0", "_ok": False,
                    "_error": f"crash: {rec['_exc']}",
                }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if rec.get("_ok"):
                n_ok += 1
            else:
                n_fail += 1
            print(f"[{i}/{len(questions)}] Q{q['id']}: ans={rec.get('answer')} err={(rec.get('_error') or '')[:60]}", flush=True)

    pipeline.close()
    print(f"\nXong. ok={n_ok} fail={n_fail} | out={out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
