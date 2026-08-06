"""prompt.py — dựng messages cho LLM Text-to-Pandas từ top-k bảng truy hồi.

Output của LLM là 1 khối code Python gán `result` (float). Code chạy trong sandbox
**khớp grader BTC** (`answering/sandbox.py`): các DataFrame nằm trong **dict `dfs`
keyed theo tên biến** — truy cập `dfs["df1"]`, `dfs["df2"]`... (1 bảng thì cũng dùng
`dfs["df1"]`, hoặc alias `df`). CSV đọc `dtype=str, index_col=None` → cells string,
index numeric. Schema tidy cố định 4 cột từ wide table M1.

Chiến lược prompt:
- System: data contract chặt (số VN, đơn vị, dfs dict, không import, gán result).
- User: câu hỏi + đơn vị yêu cầu + các bảng (mỗi bảng: meta + columns + sample rows)
  + gợi ý mã số↔nhãn từ facts (label chính xác để filter).
- Few-shot 2 ví dụ (tra cứu + đổi đơn vị; filter + chọn cột period).
"""

from __future__ import annotations

from typing import Any

from vifinqa.retrieval.entity import Entities

# ---------------------------------------------------------------------------
# System prompt — data contract + quy tắc codegen (khớp grader BTC)
# ---------------------------------------------------------------------------

_SYSTEM = """Bạn là chuyên gia viết pandas query trên bảng Báo cáo tài chính (BCTC) Việt Nam.
Các DataFrame ĐÃ CÓ DỮ LIỆU THẬT nằm trong **dict `dfs`** keyed theo tên biến:
`dfs["df1"]`, `dfs["df2"]`, ... Mỗi giá trị = 1 bảng BCTC đã chuẩn hoá với schema
cố định (đúng 4 cột, theo thứ tự):
  - "chi_tieu": tên chỉ tiêu (vd "Doanh thu thuần bán hàng và cung cấp dịch vụ", "Tổng tài sản")
  - "Mãsố": mã số chỉ tiêu (vd "10", "60"; có thể rỗng "")
  - "ky": năm kỳ báo cáo dạng chuỗi (vd "2023", "2022")
  - "value": giá trị SỐ (float), đơn vị VNĐ

QUY TẮC BẮT BUỘC:
1. Truy cập bảng qua `dfs["df1"]`, `dfs["df2"]`... TUYỆT ĐỐI CẤM tạo/gán lại DataFrame
   (`dfs["df1"] = pd.DataFrame(...)`, `= {...}`) — chỉ ĐỌC (lọc, chọn cột, tính toán).
   Nếu không thấy dữ liệu, hãy filter khác, đừng bịa. Có thể gán alias tạm: `df1 = dfs["df1"]`.
2. KHÔNG import, KHÔNG gọi pd.read_csv/open/eval/exec. Chỉ dùng dfs + pd, np, math, re.
3. Lọc dòng bằng boolean mask TRÊN CỘT. Luôn .astype(str) khi so sánh cột text (phòng dtype khác):
   - Theo mã: dfs["df1"][dfs["df1"]["Mãsố"].astype(str) == "60"]
   - Theo tên: dfs["df1"][dfs["df1"]["chi_tieu"].astype(str).str.contains("lợi nhuận sau thuế", case=False, na=False)]
   - Theo năm: dfs["df1"][dfs["df1"]["ky"].astype(str) == "2023"]
   KHÔNG dùng .index (index chỉ là số thứ tự RangeIndex), KHÔNG dùng .iloc theo vị trí cột.
4. Lấy value: dfs["df1"][...]["value"].astype(float).values[0] (hoặc .iloc[0]). Luôn ép float.
5. Kiểm tra rỗng trước khi lấy: if len(sub) == 0: result = 0.0.
6. Đổi đơn vị: value tính bằng VNĐ. "triệu đồng" → /1e6; "tỷ đồng" → /1e9; "nghìn đồng" → /1e3.
7. Tránh nhầm chỉ tiêu cha/con (vd "Lợi nhuận sau thuế" vs "Lợi nhuận sau thuế TNDN") — ưu tiên khớp Mãsố nếu có.
8. Hàm `vn_num(s)` đã có sẵn (parse số VN: 1.234,56 / (1.234) / -1234) — dùng được luôn, không cần định nghĩa lại.
9. Cuối cùng gán result = <float>. KHÔNG print. KHÔNG round trung gian.
10. Chỉ trả code trong khối ```python ... ```, không giải thích."""

_FEW_SHOT = """
VÍ DỤ 1 — tra cứu 1 giá trị + đổi đơn vị:
dfs["df1"] = BCTC thu nhập (income), cột: chi_tieu | Mãsố | ky | value (đơn vị VNĐ)
Gợi ý mã số: 60 = Lợi nhuận sau thuế TNDN, 10 = Doanh thu thuần bán hàng và cung cấp dịch vụ
Câu hỏi: "Lợi nhuận sau thuế năm 2023 của HPG là bao nhiêu tỷ đồng?"
```python
df1 = dfs["df1"]
sub = df1[(df1["Mãsố"].astype(str) == "60") & (df1["ky"].astype(str) == "2023")]
if len(sub) == 0:
    sub = df1[(df1["chi_tieu"].astype(str).str.contains("lợi nhuận sau thuế", case=False, na=False)) & (df1["ky"].astype(str) == "2023")]
result = float(sub["value"].astype(float).iloc[0]) / 1e9 if len(sub) > 0 else 0.0
```

VÍ DỤ 2 — lọc theo mã chỉ tiêu:
dfs["df1"] = BCTC cân đối kế toán (balance_sheet), cột: chi_tieu | Mãsố | ky | value
Gợi ý mã số: 270 = Tổng tài sản ngắn hạn, 440 = Tổng tài sản
Câu hỏi: "Tổng tài sản cuối năm 2022 của VCB là bao nhiêu đồng?"
```python
df1 = dfs["df1"]
sub = df1[(df1["Mãsố"].astype(str) == "440") & (df1["ky"].astype(str) == "2022")]
if len(sub) == 0:
    sub = df1[(df1["chi_tieu"].astype(str).str.contains("tổng tài sản", case=False, na=False)) & (df1["ky"].astype(str) == "2022")]
result = float(sub["value"].astype(float).iloc[0]) if len(sub) > 0 else 0.0
```"""


def _format_table_card(card: dict) -> str:
    """1 bảng → text: meta + columns + sample rows."""
    var = card["var"]
    lines = [
        f'dfs["df{var}"] = {card["report_id"]} | bảng {card["position"]} | {card["statement"] or "thuyết minh"}',
        f"  schema: cột {card['columns']} — value đơn vị VNĐ (đã chuẩn hoá)",
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

    - `table_cards`: list[dict] với keys: var (int), report_id, position, statement,
      unit, unit_factor, columns (list[str]), fact_hints (list[(code,label)] | None),
      sample_rows (str | None — đã cắt ngắn, nhiều dòng).
    """
    unit_line = "không chỉ định (để VND)"
    if entities.unit_label:
        unit_line = f"{entities.unit_label} (factor={entities.unit_factor:g})"
    yr = ", ".join(map(str, sorted(entities.years))) or "(không rõ năm — suy luận từ cột kỳ)"

    cards_text = "\n\n".join(_format_table_card(c) for c in table_cards) if table_cards else "(không có bảng truy hồi)"
    n_tables = len(table_cards)

    user = f"""Câu hỏi: {question}
Năm trong câu hỏi: {yr}
Đơn vị đáp án yêu cầu: {unit_line}

Có ĐÚNG {n_tables} bảng được cung cấp trong dict `dfs`: dfs["df1"] .. dfs["df{n_tables}"].
⚠️ CHỈ được tham chiếu dfs["df1"] .. dfs["df{n_tables}"]. KHÔNG tồn tại dfs["df{n_tables + 1}"] trở lên — tham chiếu quá số bảng sẽ lỗi KeyError.

Các bảng truy hồi (top-k):
{cards_text}

Viết 1 khối pandas query gán `result` (float) theo quy tắc. Chỉ trả code trong ```python ... ```."""

    return [
        {"role": "system", "content": _SYSTEM + "\n" + _FEW_SHOT},
        {"role": "user", "content": user},
    ]