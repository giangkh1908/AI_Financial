import json
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

recs = [json.loads(l) for l in open('data/out/results_100.jsonl', encoding='utf-8')]

# Phân tích pandas_query: df nào được tham chiếu?
df_usage = []
for r in recs:
    q = r.get('pandas_query', '')
    # tìm df1..dfN được dùng (không phải trong comment/string đơn giản)
    refs = sorted(set(re.findall(r'\bdf(\d+)\b', q)), key=int)
    refs = ['df' + x for x in refs]
    n_ev = len(r.get('evidence', []))
    used = [ev['variable'] for ev in r.get('evidence', []) if ev['variable'] in set(refs)]
    df_usage.append((r['id'], n_ev, refs, r.get('_error', '')[:20]))

# Thống kê
n_evs = [x[1] for x in df_usage]
n_used = [len(x[2]) for x in df_usage]
import statistics
print("evidence count: median={} mean={:.1f}".format(statistics.median(n_evs), sum(n_evs)/len(n_evs)))
print("df refs trong query: median={} mean={:.1f}".format(statistics.median(n_used), sum(n_used)/len(n_used)))

# Bao nhiêu record dùng ít bảng hơn evidence?
fewer = sum(1 for a, b, c, e in df_usage if len(c) < b)
print("query dùng FEWER df so với evidence:", fewer, f"({fewer/len(df_usage):.0%})")

# Với deterministic: query template dùng 1 bảng
det = [x for x in df_usage if x[3].startswith('deterministic')]
print("\ndeterministic:", len(det), "| median df refs:", statistics.median(len(x[2]) for x in det))

# Vd vài record
print("\n=== Ví dụ ===")
for r in recs[:5]:
    q = r.get('pandas_query', '')
    refs = sorted(set(re.findall(r'\bdf(\d+)\b', q)), key=int)
    ev = [e['variable'] for e in r.get('evidence', [])]
    print("Q{}: evidence={} query_dfs={} rt_tables={}".format(
        r['id'], len(ev), len(refs), len(r.get('relevant_tables', []))))
