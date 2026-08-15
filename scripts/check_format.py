import json
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

subs = json.load(open('data/out/submission_full/submission.json', encoding='utf-8'))
data_dir = 'data/out/submission_full/data'

REQ_FIELDS = {"id", "question", "answer", "relevant_docs", "relevant_tables", "evidence", "pandas_query"}

# 1. Format kiểm tra
bad_fields = []
bad_types = []
bad_tables = []
bad_csv = []
missing_csv = []
zero_query = []

for s in subs:
    if set(s.keys()) != REQ_FIELDS:
        bad_fields.append(s['id'])
    if not isinstance(s['id'], int) or not isinstance(s['question'], str) or not isinstance(s['answer'], float):
        bad_types.append(s['id'])
    if isinstance(s['answer'], float) and s['answer'] != s['answer']:  # NaN
        bad_types.append(s['id'])
    # relevant_tables format: report_id|<number>
    for rt in s.get('relevant_tables', []):
        if not re.fullmatch(r'[^|]+\|\d+', rt):
            bad_tables.append((s['id'], rt))
    # evidence csv_path + variable unique
    seen_var = set()
    for ev in s.get('evidence', []):
        v = ev.get('variable', '')
        p = ev.get('csv_path', '')
        if v in seen_var or not re.fullmatch(r'[A-Za-z_]\w*', v):
            bad_csv.append((s['id'], 'var', v))
        seen_var.add(v)
        if not p.startswith('data/') or '..' in p:
            bad_csv.append((s['id'], 'path', p))
        else:
            rel = p[len('data/'):]
            if not os.path.exists(os.path.join(data_dir, rel)):
                missing_csv.append((s['id'], p))
    if s.get('pandas_query', '').strip() in ('', 'result = 0.0', 'result=0.0'):
        zero_query.append(s['id'])

print("=== Format ===")
print("Tổng record:", len(subs))
print("Sai field:", bad_fields[:5], "| sai type:", bad_types[:5])
print("relevant_tables sai format:", len(bad_tables), bad_tables[:3])
print("evidence sai var/path:", len(bad_csv), bad_csv[:3])
print("csv_path thiếu file:", len(missing_csv), missing_csv[:3])
print("pandas_query rỗng/0.0:", len(zero_query))

# 2. Thống kê chất lượng
total = len(subs)
zero = [s for s in subs if s['answer'] == 0.0]
print("\n=== Chất lượng ===")
print("answer=0.0:", len(zero), f"({len(zero)/total:.1%})")
print("answer!=0:", total - len(zero))

# 3. relevant_docs vs relevant_tables: đếm bảng trung bình
rt_counts = [len(s.get('relevant_tables', [])) for s in subs]
print("\nrelevant_tables: median={} max={} >10 câu={}".format(
    sorted(rt_counts)[len(rt_counts)//2], max(rt_counts), sum(1 for x in rt_counts if x > 10)))

# 4. Đếm evidence trung bình
ev_counts = [len(s.get('evidence', [])) for s in subs]
print("evidence: median={} max={}".format(sorted(ev_counts)[len(ev_counts)//2], max(ev_counts)))
