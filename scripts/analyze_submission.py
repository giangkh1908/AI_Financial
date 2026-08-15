import json
import sys
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

recs = [json.loads(l) for l in open('data/out/results_100.jsonl', encoding='utf-8')]
print("Total:", len(recs))

# Phân bố phương pháp
methods = Counter()
for r in recs:
    err = r.get('_error', '')
    if err.startswith('deterministic'):
        methods['deterministic'] += 1
    elif 'không có bảng truy hồi' in err:
        methods['no_evidence'] += 1
    elif 'LLM không trả code' in err:
        methods['llm_no_code'] += 1
    elif 'wall-cap' in err:
        methods['wallcap'] += 1
    elif 'ast_check' in err or 'dfs[' in err:
        methods['ast_rejected'] += 1
    elif err:
        methods[f'other: {err[:30]}'] += 1
    else:
        methods['llm_ok'] += 1

print("\n=== Phương pháp ===")
for k, v in methods.most_common():
    print(f"  {k}: {v}")

# Answer 0.0 distribution
zero = [r for r in recs if r['answer'] == 0.0]
print(f"\nanswer=0.0: {len(zero)}/{len(recs)}")
nonzero = [r for r in recs if r['answer'] != 0.0]
print(f"answer!=0:  {len(nonzero)}/{len(recs)}")

# Evidence count
ev_counts = [len(r.get('evidence', [])) for r in recs]
import statistics
print(f"\nevidence count: min={min(ev_counts)} max={max(ev_counts)} median={statistics.median(ev_counts)}")
rt_counts = [len(r.get('relevant_tables', [])) for r in recs]
print(f"relevant_tables: min={min(rt_counts)} max={max(rt_counts)} median={statistics.median(rt_counts)}")

# Kiểm tra xem có câu nào evidence 0 nhưng answer khác 0 (nghi vấn)
noev = [r for r in recs if not r.get('evidence')]
print(f"\nevidence=0: {len(noev)}")
for r in noev[:5]:
    print(f"  Q{r['id']}: ans={r['answer']} err={r.get('_error','')[:60]}")
