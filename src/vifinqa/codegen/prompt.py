"""prompt.py — dựng messages cho LLM Text-to-Pandas từ top-k bảng truy hồi.

Output của LLM là 1 khối code Python gán `result` (float). Code chạy trong sandbox
**khớp grader BTC** (`answering/sandbox.py`):
- Các DataFrame nằm trong **dict `dfs` keyed theo `table_ref`** =
  `"{report_id}|table_{N}"`. Khi chỉ 1 bảng, grader thêm alias **`df`**.
- Namespace CHỈ có `pd` + builtins (abs, round, len, min, max, sum, sorted, float,
  int, str, bool, list, dict, set, range, enumerate, zip, all, any, isinstance).
  KHÔNG có np/math/re/json. Có sẵn helper `vn_num` (parse số VN).
- CSV đọc `dtype=str, index_col=None` → cells string, index numeric RangeIndex.
- `result` = scalar, **làm tròn 2 chữ số thập phân**.

Chiến lược prompt:
- System: data contract chặt (số VN, đơn vị, dfs[df]/df, chỉ pd+builtins, round 2).
- User: câu hỏi + đơn vị + các bảng (mỗi bảng: table_ref + meta + columns + sample
  rows + gợi ý mã số↔nhãn).
- Few-shot 2 ví dụ (1 bảng dùng df; N bảng dùng dfs["<table_ref>"]).
"""

from __future__ import annotations

from typing import Any

from vifinqa.retrieval.entity import Entities

# ---------------------------------------------------------------------------
# System prompt — data contract + quy tắc codegen (khớp grader BTC)
# ---------------------------------------------------------------------------

_SYSTEM = """Bạn là chuyên gia viết pandas query trên bảng Báo cáo tài chính (BCTC) Việt Nam.

⚠️ QUY TẮC QUAN TRỌNG NHẤT — ĐỌC KỸ:
Mỗi bảng trong evidence có một VARIABLE NAME (df1, df2, df3, ...). Bạn PHẢI dùng
CHÍNH XÁC các variable names này trong query. TUYỆT ĐỐI CẤM dùng `dfs["..."]` hoặc
tự bịa variable names khác.

VÍ DỤ ĐÚNG:   t = df1  (nếu evidence ghi variable=df1)
VÍ DỤ SAI:    t = dfs["VJC_...|table_9"]  (SAI — dfs không tồn tại)

MÔI TRƯỜNG (đã có sẵn, KHÔNG cần import/khởi tạo):
- `pd` (pandas). Builtins cho phép: abs, round, len, min, max, sum, sorted, float,
  int, str, bool, list, dict, set, range, enumerate, zip, all, any, isinstance.
  KHÔNG có numpy/math/re/json — đừng dùng.
- `vn_num(s)`: parse chuỗi số VN → float (vd "1.234,56"→1234.56, "(1.234)"→-1234,
  "-1234"→-1234, ""/"-"→None). Đã có sẵn, dùng luôn, KHÔNG cần định nghĩa lại.
- Các DataFrames đã có sẵn với tên từ evidence: `df1`, `df2`, `df3`, ... (1 bảng cũng
  có thể dùng `df1` hoặc `df`). Dùng CHÍNH XÁC tên variable từ evidence.

SCHEMA mỗi bảng (đúng 4 cột, theo thứ tự):
  - "chi_tieu": tên chỉ tiêu (vd "Doanh thu thuần bán hàng và cung cấp dịch vụ")
  - "Mãsố": mã số chỉ tiêu (vd "10", "60"; có thể rỗng "")
  - "ky": năm kỳ báo cáo dạng chuỗi (vd "2023", "2022")
  - "value": giá trị dạng chuỗi (dtype=str) → parse bằng `float()` hoặc `vn_num()`,
    đơn vị VNĐ (đã chuẩn hoá).

QUY TẮC BẮT BUỘC:
1. Truy cập bảng: Dùng CHÍNH XÁC variable names từ evidence (df1, df2, df3, ...).
   TUYỆT ĐỐI CẤM tạo/gán lại DataFrame (`df1 = pd.DataFrame(...)`).
   Chỉ ĐỌC (lọc, chọn cột, tính toán). Có thể gán alias tạm `t = df1`.
   TUYỆT ĐỐI CẤM `dfs["..."]` (dfs không tồn tại).
2. KHÔNG import, KHÔNG gọi pd.read_csv/open/eval/exec. Chỉ dùng pd + builtins + vn_num.
3. Lọc dòng bằng boolean mask TRÊN CỘT. Luôn .astype(str) khi so sánh cột text:
   - Theo mã:   df[df["Mãsố"].astype(str) == "60"]
   - Theo tên:  df[df["chi_tieu"].astype(str).str.contains("lợi nhuận sau thuế", case=False, na=False)]
   - Theo năm:  df[df["ky"].astype(str) == "2023"]
   KHÔNG dùng .index (chỉ là số thứ tự RangeIndex), KHÔNG .iloc theo vị trí cột.
4. Lấy value: `float(sub["value"].iloc[0])` (hoặc `vn_num(sub["value"].iloc[0])`).
   Cột "value" là chuỗi → LUÔN ép float trước khi tính toán.
5. Kiểm tra rỗng: `if len(sub) == 0: result = 0.0`.
6. Đổi đơn vị: value tính bằng VNĐ. "triệu đồng"→/1e6; "tỷ đồng"→/1e9; "nghìn đồng"→/1e3.
7. Tránh nhầm chỉ tiêu cha/con — ưu tiên khớp "Mãsố" nếu có.
8. Cuối cùng gán `result = round(<float>, 2)`. KHÔNG print. KHÔNG round trung gian.
9. KHÔNG viết comment (`# ...`) trong code — code ngắn gọn, chỉ có logic. Comment làm
   đầy token và dễ cắt code. Chỉ trả code trong khối ```python ... ```, không giải thích."""

_FEW_SHOT = """
VÍ DỤ 1 — 1 bảng (variable=df1), tra cứu + đổi đơn vị:
df1 = BCTC thu nhập (income), schema: chi_tieu | Mãsố | ky | value (VNĐ, chuỗi)
Gợi ý mã số: 60 = Lợi nhuận sau thuế TNDN, 10 = Doanh thu thuần
Câu hỏi: "Lợi nhuận sau thuế năm 2023 của HPG là bao nhiêu tỷ đồng?"
```python
sub = df1[(df1["Mãsố"].astype(str) == "60") & (df1["ky"].astype(str) == "2023")]
if len(sub) == 0:
    sub = df1[(df1["chi_tieu"].astype(str).str.contains("lợi nhuận sau thuế", case=False, na=False)) & (df1["ky"].astype(str) == "2023")]
result = round(float(sub["value"].iloc[0]) / 1e9, 2) if len(sub) > 0 else 0.0
```

VÍ DỤ 2 — N bảng (variable=df1, df2), so sánh 2 năm tăng trưởng:
Các bảng có sẵn:
  df1 = BCTC thu nhập 2023, schema: chi_tieu | Mãsố | ky | value
  df2 = BCTC thu nhập 2022, schema: chi_tieu | Mãsố | ky | value
Gợi ý mã số: 10 = Doanh thu thuần
Câu hỏi: "Doanh thu thuần 2023 của HPG tăng bao nhiêu phần trăm so với 2022?"

⚠️ CHÚ Ý: Dùng variable names df1, df2 từ evidence. KHÔNG dùng dfs["..."].

```python
sub_2023 = df1[(df1["Mãsố"].astype(str) == "10") & (df1["ky"].astype(str) == "2023")]
sub_2022 = df2[(df2["Mãsố"].astype(str) == "10") & (df2["ky"].astype(str) == "2022")]
if len(sub_2023) == 0 or len(sub_2022) == 0:
    result = 0.0
else:
    a = float(sub_2023["value"].iloc[0])
    b = float(sub_2022["value"].iloc[0])
    result = round((a - b) / b * 100, 2) if b != 0 else 0.0
```"""


def _format_table_card(card: dict) -> str:
    """1 bảng → text: variable name + meta + columns + sample rows."""
    # card có key "var_name" (df1, df2, ...) hoặc fallback về table_ref
    var_name = card.get("var_name", card["table_ref"])
    lines = [
        f'{var_name} = {card["report_id"]} | bảng {card["position"]} | {card["statement"] or "thuyết minh"}',
        f"  schema: cột {card['columns']} — value chuỗi (dtype=str), đơn vị VNĐ (đã chuẩn hoá)",
        f"  gốc đơn vị báo cáo: {card['unit'] or 'VND'} (unit_factor={card['unit_factor']})",
    ]
    if card.get("fact_hints"):
        hints = "; ".join(f"{c} = {lab}" for c, lab in card["fact_hints"][:25])
        lines.append(f"  gợi ý mã số: {hints}")
    if card.get("sample_rows"):
        lines.append("  mẫu dữ liệu (chi_tieu | Mãsố | ky | value):")
        for row in card["sample_rows"].splitlines():
            lines.append(f"    {row}")
    return "\n".join(lines)


def build_messages(
    question: str,
    entities: Entities,
    table_cards: list[dict[str, Any]],
) -> list[dict]:
    """Danh sách messages cho LLM.

    - `table_cards`: list[dict] với keys: table_ref (str), report_id, position,
      statement, unit, unit_factor, columns (list[str]), fact_hints, sample_rows.
    """
    unit_line = "không chỉ định (để VND)"
    if entities.unit_label:
        unit_line = f"{entities.unit_label} (factor={entities.unit_factor:g})"
    yr = ", ".join(map(str, sorted(entities.years))) or "(không rõ năm — suy luận từ cột kỳ)"

    cards_text = "\n\n".join(_format_table_card(c) for c in table_cards) if table_cards else "(không có bảng truy hồi)"
    n_tables = len(table_cards)

    if n_tables == 1:
        access_rule = (
            "Có 1 bảng → DataFrame có sẵn tên `df1`. Dùng `df1` trong query. "
            "TUYỆT ĐỐI CẤM dùng `dfs[\"...\"]` (dfs không tồn tại)."
        )
    elif n_tables > 1:
        var_names = ", ".join(f"df{i+1}" for i in range(n_tables))
        table_info = "\n".join(
            f"  df{i+1} = {c['report_id']} | bảng {c['position']} | {c['statement'] or 'thuyết minh'}"
            for i, c in enumerate(table_cards)
        )
        access_rule = (
            f"⚠️ CÓ {n_tables} BẢNG — ĐỌC KỸ:\n"
            f"Các variable names có sẵn: {var_names}\n"
            f"Thông tin bảng:\n{table_info}\n"
            f"BẠN PHẢI dùng CHÍNH XÁC các variable names trên (df1, df2, ...).\n"
            f"TUYỆT ĐỐI CẤM: `dfs[\"...\"]` — dfs KHÔNG TỒN TẠI.\n"
            f"VÍ DỤ ĐÚNG: `t = df1` hoặc `t = df2`\n"
            f"VÍ DỤ SAI: `t = dfs[\"df1\"]` hoặc `t = dfs[\"VJC_...|table_9\"]` → KeyError"
        )
    else:
        access_rule = "Không có bảng → `result = 0.0`."

    user = f"""Câu hỏi: {question}
Năm trong câu hỏi: {yr}
Đơn vị đáp án yêu cầu: {unit_line}

{access_rule}

Các bảng truy hồi (top-k):
{cards_text}

Viết 1 khối pandas query gán `result` (float, round 2 chữ số thập phân) theo quy tắc. Chỉ trả code trong ```python ... ```."""

    return [
        {"role": "system", "content": _SYSTEM + "\n" + _FEW_SHOT},
        {"role": "user", "content": user},
    ]