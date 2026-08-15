import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open('data/out/submission_full/submission.json', encoding='utf-8') as f:
    subs = json.load(f)

print('So record:', len(subs))
for s in subs[:3]:
    rt = s.get('relevant_tables', [])
    print("Q{}: {} tables | vd: {}".format(s['id'], len(rt), rt[:3]))

table_N = sum(1 for s in subs for r in s.get('relevant_tables', []) if '|table_' in r)
start_line = sum(1 for s in subs for r in s.get('relevant_tables', []) if '|table_' not in r and '|' in r)
print('table_N format:', table_N, '| start_line format:', start_line)

# Kiểm tra id câu
ids = [s['id'] for s in subs]
print('IDs min/max:', min(ids), max(ids), 'unique:', len(set(ids)))
