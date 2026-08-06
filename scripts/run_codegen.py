"""run_codegen.py — batch codegen toàn bộ questions → data/out/results.jsonl.

Checkpoint theo id (resume bỏ qua id đã có trong file). Concurrency 4 (configurable
qua --workers). Guard lỗi: lỗi 1 câu không brick batch.

Chạy:
  python scripts/run_codegen.py --limit 10        # smoke 10 câu
  python scripts/run_codegen.py --limit 0         # full 1012
  python scripts/run_codegen.py --limit 0 --workers 8
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _load_done(out_path: Path) -> set[int]:
    if not out_path.exists():
        return set()
    done: set[int] = set()
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10, help="số câu (0 = full)")
    ap.add_argument("--workers", type=int, default=4, help="concurrency")
    ap.add_argument("--config", default=str(ROOT / "configs" / "api.yaml"))
    ap.add_argument("--out", default=None, help="path results.jsonl (mặc định data/out/results.jsonl)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute() and not cfg_path.exists():
        cfg_path = ROOT / "configs" / args.config
    cfg = Config.load(cfg_path)
    out_dir = cfg.resolved_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / "results.jsonl"

    questions = load_questions(cfg.resolved_data_dir() / "questions" / "questions.jsonl")
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    done = _load_done(out_path)
    pending = [q for q in questions if q["id"] not in done]
    print(f"Questions: {len(questions)} | đã xong: {len(done)} | còn lại: {len(pending)}", flush=True)
    if not pending:
        print("Không còn câu cần chạy.")
        return 0

    pipeline = RetrievalPipeline(cfg)
    facts_index = FactsIndex(cfg.resolved_derived_dir() / "facts_all.csv")
    llm = LLMClient(cfg)

    t0 = time.time()
    n_ok = n_fail = n_retry = 0
    latencies: list[float] = []

    WALL_CAP = float(cfg.sandbox.timeout) * 6.0  # 20s pandas ×6 = 120s/question

    def _do(q):
        ts = time.time()
        qres: queue.Queue = queue.Queue()

        def _worker():
            try:
                qres.put(solve(q["question"], q["id"], pipeline, facts_index, llm, cfg))
            except Exception as e:  # guard lỗi — không brick batch
                qres.put({"_exc": e})

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        try:
            rec = qres.get(timeout=WALL_CAP)
        except queue.Empty:
            rec = None  # vượt wall-cap → fallback
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
        return rec, time.time() - ts

    # append mode: ghi từng record ngay khi xong (đồng bộ ghi để an toàn)
    lock_file = out_path.with_suffix(".lock")
    with open(out_path, "a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_do, q): q for q in pending}
            for fut in as_completed(futs):
                try:
                    rec, lat = fut.result()
                except Exception as e:
                    q = futs[fut]
                    rec = {
                        "id": q["id"], "question": q["question"], "answer": 0.0,
                        "relevant_docs": [], "relevant_tables": [], "evidence": [],
                        "pandas_query": "result = 0.0", "_ok": False, "_error": f"crash: {e}",
                    }
                    lat = 0.0
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                latencies.append(lat)
                if rec.get("_ok"):
                    n_ok += 1
                else:
                    n_fail += 1
                if rec.get("_error"):
                    n_retry += 1
                done_n = n_ok + n_fail
                if done_n % 10 == 0 or done_n == len(pending):
                    p95 = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0
                    print(
                        f"[{done_n}/{len(pending)}] ok={n_ok} fail={n_fail} "
                        f"p95={p95:.1f}s elapsed={time.time() - t0:.0f}s",
                        flush=True,
                    )

    pipeline.close()
    print(f"\nXong. ok={n_ok} fail={n_fail} (fallback) | total={time.time() - t0:.0f}s | out={out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())