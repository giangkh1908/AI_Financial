"""run_codegen_spots.py — chạy codegen cho 1 tập câu cụ thể (spot ids) → results riêng.

Dùng để test nhanh ảnh hưởng của config retrieval (dense-only, k, bonus...) mà
không phải chạy full 1012. Ghi kết quả vào `data/out/results_spots.jsonl`.

Chạy:
  python scripts/run_codegen_spots.py --ids 1,11,15,21,30,369,378,400,689,790 --out data/out/results_spots_denseonly.jsonl
"""

from __future__ import annotations

import argparse
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
from vifinqa.agent.loop import solve  # noqa: E402
from vifinqa.loader import load_questions  # noqa: E402
from vifinqa.retrieval.facts_index import FactsIndex  # noqa: E402
from vifinqa.retrieval.pipeline import RetrievalPipeline  # noqa: E402
from vifinqa.codegen.llm import LLMClient  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="api.yaml")
    ap.add_argument("--ids", default="1,11,15,21,30,369,378,400,689,790", help="danh sách id cách dấu phẩy")
    ap.add_argument("--out", default="data/out/results_spots.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = Config.load(ROOT / "configs" / args.config)
    spot = [int(x) for x in args.ids.split(",") if x]

    questions = load_questions(cfg.resolved_data_dir() / "questions" / "questions.jsonl")
    qmap = {q["id"]: q for q in questions}
    chosen = [qmap[i] for i in spot if i in qmap]
    print(f"Chạy {len(chosen)} câu spot {spot} (workers={args.workers})...", flush=True)

    pipeline = RetrievalPipeline(cfg)
    facts_index = FactsIndex(cfg.resolved_derived_dir() / "facts_all.csv")
    llm = LLMClient(cfg)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            line = line.strip()
            if line:
                done.add(json.loads(line)["id"])
    pending = [q for q in chosen if q["id"] not in done]
    print(f"Đã xong: {len(done)} | còn chạy: {len(pending)}", flush=True)

    t0 = time.time()
    try:
        for q in pending:
            ts = time.time()
            rec = solve(q["question"], q["id"], pipeline, facts_index, llm, cfg)
            dt = time.time() - ts
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            err = (rec.get("_error") or "")[:40]
            print(f"  Q{q['id']}: ans={rec['answer']} ({dt:.0f}s) err={err}", flush=True)
    finally:
        pipeline.close()

    print(f"\nDone trong {time.time()-t0:.0f}s → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())