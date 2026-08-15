import json
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

results = []
with open('data/out/results_test20.jsonl', encoding='utf-8') as f:
    for line in f:
        results.append(json.loads(line))

llm_count = 0
det_count = 0
for r in results:
    err = r.get('_error', '')
    qid = r['id']
    if 'deterministic' in err:
        det_count += 1
    else:
        llm_count += 1
        pq = r.get('pandas_query', '')[:80]
        print(f'Q{qid}: LLM | ans={r["answer"]} | {pq}...')

print(f'\n--- Summary 20 cau ---')
print(f'Deterministic: {det_count}')
print(f'LLM codegen:   {llm_count}')
