"""smoke_fmt_100.py — build submission 100 câu (subset) để kiểm tra format + zip.

Lấy 100 record đầu từ results.jsonl (ưu tiên câu có answer≠0 cho thực tế),
match questions tương ứng, build → validate → pack, rồi in kiểm tra format.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.config import Config  # noqa: E402
from vifinqa.submission import build, pack, validate  # noqa: E402


def main() -> int:
    cfg = Config.load(ROOT / "configs" / "api.yaml")
    out_dir = ROOT / "data" / "out" / "submission_100"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Lấy 100 record (ưu tiên answer≠0), giữ id
    recs_all = []
    with (ROOT / "data" / "out" / "results.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs_all.append(json.loads(line))
    nonzero = [r for r in recs_all if r.get("answer", 0.0) != 0.0][:100]
    if len(nonzero) < 100:
        nonzero = recs_all[:100]
    ids_100 = {r["id"] for r in nonzero}
    print(f"Lấy {len(nonzero)} record (answer≠0 ưu tiên), id sample: {sorted(ids_100)[:5]}")

    # 2. Match questions
    qs_all = []
    with (cfg.resolved_data_dir() / "questions" / "questions.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qs_all.append(json.loads(line))
    qs_100 = [q for q in qs_all if q["id"] in ids_100]
    qpath = ROOT / "data" / "out" / "questions_100.jsonl"
    with qpath.open("w", encoding="utf-8") as f:
        for q in qs_100:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # 3. Build
    summ = build(ROOT / "data" / "out" / "results.jsonl", out_dir, cfg.resolved_derived_dir(), qpath)
    print(f"[build] n={summ['n']} materialized={summ['materialized']} "
          f"tidy_regen={summ.get('tidy_regen',0)} rt_rewritten={summ.get('relevant_tables_rewritten',0)} "
          f"missing={len(summ['missing_ids'])}")

    # 4. Validate
    vsumm = validate(
        out_dir / "submission.json", out_dir / "data", out_dir / "validation.jsonl",
        abs_tol=cfg.answer_abs_tol, timeout=cfg.sandbox.timeout,
        max_code_len=cfg.sandbox.max_code_len, max_ast_nodes=cfg.sandbox.max_ast_nodes,
    )
    print(f"[validate] {json.dumps(vsumm)}")

    # 5. Pack
    zip_path = pack(out_dir)
    print(f"[pack] {zip_path} ({zip_path.stat().st_size / 1024:.0f} KB)")

    # 6. Format check
    print("\n=== FORMAT CHECK ===")
    recs = json.loads((out_dir / "submission.json").read_text(encoding="utf-8"))
    print(f"records: {len(recs)} (spec: dự đoán cho toàn bộ câu hỏi trong subset = {len(qs_100)})")
    keys_expected = {"id", "question", "answer", "relevant_docs", "relevant_tables", "evidence", "pandas_query"}
    r0 = recs[0]
    print(f"record[0] keys: {set(r0.keys())} | match spec: {set(r0.keys()) == keys_expected}")
    print(f"  id={r0['id']} (type {type(r0['id']).__name__}) | answer={r0['answer']} (type {type(r0['answer']).__name__})")
    print(f"  relevant_docs sample: {r0['relevant_docs'][:2]}")
    print(f"  relevant_tables sample: {r0['relevant_tables'][:3]}")
    rt_ok = all(isinstance(t, str) and "|" in t and t.split("|",1)[1].lstrip("table_").isdigit() for t in r0["relevant_tables"])
    # check line-format: second part should be a pure integer (line number)
    def is_line_fmt(t):
        if not isinstance(t, str) or "|" not in t: return False
        right = t.split("|",1)[1]
        return right.isdigit()
    line_fmt = all(is_line_fmt(t) for t in r0["relevant_tables"])
    print(f"  relevant_tables ALL line-format (rid|<int>): {line_fmt}")
    print(f"  evidence sample: {r0['evidence'][:2]}")
    ev_ok = all("variable" in e and "csv_path" in e and e["csv_path"].startswith("data/") for e in r0["evidence"])
    print(f"  evidence ALL schema ok: {ev_ok}")
    print(f"  pandas_query head: {r0['pandas_query'][:80]!r}")

    # 7. ZIP structure check
    print("\n=== ZIP CHECK ===")
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        n_json = [n for n in names if n.endswith(".json")]
        n_csv = [n for n in names if n.endswith(".csv")]
        root_json = [n for n in names if n == "submission.json"]
        root_data = [n for n in names if n.startswith("data/")]
        nested = [n for n in names if "/" in n and not (n.startswith("data/") or n == "submission.json")]
        print(f"total entries: {len(names)} | .json files: {len(n_json)} (spec: exactly 1) | .csv: {len(n_csv)}")
        print(f"submission.json at root: {len(root_json)==1}")
        print(f"data/ entries: {len(root_data)}")
        print(f"nested in other folder (spec: none): {nested[:3]}")
        # check a csv_path from submission.json exists in zip
        sample_csv = r0["evidence"][0]["csv_path"] if r0["evidence"] else None
        if sample_csv:
            print(f"sample csv_path '{sample_csv}' in zip: {sample_csv in names}")

    print("\n=== sample record (full) ===")
    print(json.dumps(r0, ensure_ascii=False, indent=2)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())