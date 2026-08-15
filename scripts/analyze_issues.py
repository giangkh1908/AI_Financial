import json
import sys
import re
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

recs = [json.loads(l) for l in open('data/out/results_100.jsonl', encoding='utf-8')]

# 1. Câu hỏi phần trăm / tỷ lệ - nghi ngờ sai đơn vị
pct_questions = [r for r in recs if re.search(r'ph[âa]n tr[ăa]m|t[ỷy] l[ệe]|%', r['question'])]
print(f"=== Câu hỏi % / tỷ lệ: {len(pct_questions)} câu ===")
big = [r for r in pct_questions if abs(r['answer']) > 1000]
print(f"  - trả về số > 1000 (nghi ngờ sai đơn vị): {len(big)}")
for r in big[:8]:
    print(f"  Q{r['id']}: ans={r['answer']:,.0f} | {r['question'][:70]}")
small = [r for r in pct_questions if 0 < abs(r['answer']) <= 100]
print(f"  - trả về số hợp lý cho % (0-100): {len(small)}")

# 2. Unit factor: câu hỏi "triệu" nhưng trả số khổng lồ
print("\n=== Phân tích answer theo magnitude ===")
neg = [r for r in recs if r['answer'] < 0]
print(f"  answer âm: {len(neg)}")
huge = [r for r in recs if abs(r['answer']) >= 1e12]
print(f"  answer >= 1e12 (hàng nghìn tỷ): {len(huge)}")
for r in huge[:5]:
    print(f"  Q{r['id']}: ans={r['answer']:.0f} | {r['question'][:60]}")

# 3. So sánh: deterministic vs llm (không có gold, chỉ xem phân bố)
print("\n=== Deterministic: phân bố answer magnitude ===")
det = [r for r in recs if r.get('_error','').startswith('deterministic')]
zero_det = [r for r in det if r['answer']==0.0]
print(f"  deterministic total={len(det)}, answer=0: {len(zero_det)}")
det_ok = [r for r in det if r['answer']!=0.0]
# unit: nếu câu hỏi triệu nhưng answer >= 1e9 -> sai đơn vị x1000
mismatch = 0
for r in det_ok:
    q = r['question']
    if 'triệu đồng' in q and abs(r['answer']) >= 1e6:
        mismatch += 1
print(f"  deterministic nonzero={len(det_ok)}, câu hỏi 'triệu' mà answer>=1e6 (sai x1000?): {mismatch}")

# 4. Xem relevant_tables có quá nhiều không
rt = [len(r.get('relevant_tables', [])) for r in recs]
print(f"\n=== relevant_tables count: median={sorted(rt)[len(rt)//2]}, >15: {sum(1 for x in rt if x>15)} câu ===")

# 5. Kiểm tra questions có "công ty mẹ" mà trả consolidated
print("\n=== 'công ty mẹ' vs report type ===")
for r in recs[:0]:
    pass
cm = [r for r in recs if 'công ty mẹ' in r['question'] or 'của công ty mẹ' in r['question']]
print(f"  câu có 'công ty mẹ': {len(cm)}")
