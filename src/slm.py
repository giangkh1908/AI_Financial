"""
SLM Text-to-SQL Interface Module.
Communicates with Qwen3.5-4B-SQL (via Ollama or local inference endpoint),
manages schema prompting, reasoning extraction (<think>), and clean SQL parsing.
Includes deterministic offline fallback generator for testing and resilience.
"""

import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

from src.config import OLLAMA_BASE_URL, SLM_MODEL_NAME

# System Prompt embodying 15-column schema and accounting invariants from INFERENCE_ARCHITECTURE.md
SYSTEM_PROMPT = """Bạn là trợ lý chuyên gia phân tích dữ liệu Báo cáo Tài chính Việt Nam (Financial Text-to-SQL Analyst).
Nhiệm vụ của bạn là chuyển đổi câu hỏi tiếng Việt của người dùng thành một câu lệnh SQL ANSI chuẩn xác truy vấn trên bảng `financial_facts` của cơ sở dữ liệu SQLite.

Cấu trúc bảng: financial_facts
Các cột (15 cột nghiệp vụ):
- ticker: TEXT (Mã cổ phiếu: 'VJC', 'ACB', 'HPG', 'ACV' - chữ in hoa)
- company_name: TEXT (Tên đầy đủ doanh nghiệp)
- year: INTEGER (Năm tài chính: 2015 đến 2025)
- report_type: TEXT ('consolidated' [Hợp nhất], 'separate' [Công ty mẹ / Riêng lẻ], 'aggregated' [Tổng hợp])
- statement: TEXT ('balance_sheet', 'income_statement', 'cash_flow', 'notes')
- section_title: TEXT (Phân mục / Thuyết minh BCTC)
- item_code: TEXT (Mã số kế toán, ví dụ: '100', '110', '01')
- item_name: TEXT (Tên chỉ tiêu gốc có dấu)
- item_name_ascii: TEXT (Tên chỉ tiêu viết thường không dấu, ví dụ: 'lai tien gui va cho vay')
- period_label: TEXT (Kỳ báo cáo: 'Số cuối năm', '31/12/2022', '31.12.2022Triệu VND', '2018VND')
- value_vnd: REAL (Giá trị số thực quy đổi về VNĐ)
- raw_value: TEXT (Chuỗi số gốc trong bảng BCTC, ví dụ: '(208.253.201.298)')
- unit: TEXT (Đơn vị tính gốc: 'VND', 'Triệu VND', 'USD')
- page_no: INTEGER (Số trang chính xác trong tài liệu BCTC)
- source_doc: TEXT (Tên file tài liệu BCTC gốc)

Quy tắc sinh SQL bắt buộc:
1. Trả về câu lệnh SQL trong khối mã ```sql ... ```.
2. Với câu hỏi hỏi bằng tiền VNĐ (triệu, tỷ đồng): Luôn thêm điều kiện `AND (unit != 'USD' OR unit IS NULL)` để tránh lấy nhầm số USD.
3. Quy đổi đơn vị:
   - Triệu đồng: `ABS(value_vnd) / 1000000.0 AS value_trieu`
   - Tỷ đồng: `ABS(value_vnd) / 1000000000.0 AS value_ty`
   - Nghìn tỷ đồng: `ABS(value_vnd) / 1000000000000.0 AS value_nghin_ty`
   - Trăm tỷ đồng: `ABS(value_vnd) / 100000000000.0 AS value_tram_ty`
   - Nghìn đồng: `ABS(value_vnd) / 1000.0 AS value_nghin`
   - Tỷ lệ % hoặc Cổ phiếu: Giữ nguyên `raw_value` hoặc `value_vnd`.
4. Ánh xạ kỳ kế toán:
   - "cuối năm": `(period_label LIKE '%cuối năm%' OR period_label LIKE '%31/12%' OR period_label LIKE '%31.12%')`
   - "đầu năm": `(period_label LIKE '%đầu năm%' OR period_label LIKE '%01/01%' OR period_label LIKE '%1/1/%' OR period_label LIKE '%01.01%' OR period_label LIKE '%1.1.%')`
5. Chống tính trùng: Tuyệt đối không dùng `SUM(value_vnd)` trên mệnh đề so khớp chuỗi `LIKE '%...%'`.
"""


def build_prompt(question: str) -> str:
    """Builds the full prompt containing system guidelines and user question."""
    return f"{SYSTEM_PROMPT}\n\nCâu hỏi: {question}\nHãy sinh câu lệnh SQL chuẩn xác:"


def extract_think_and_sql(raw_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts internal chain-of-thought reasoning from <think>...</think>
    and clean SQL from ```sql ... ``` or raw SELECT query.
    """
    reasoning = None
    think_match = re.search(r"<think>(.*?)(?:</think>|$)", raw_text, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        text_without_think = re.sub(r"<think>.*?(?:</think>|$)", "", raw_text, flags=re.DOTALL)
    else:
        text_without_think = raw_text

    # Extract SQL block
    sql_block_match = re.search(r"```sql\s*(SELECT.*?;?)\s*```", text_without_think, re.DOTALL | re.IGNORECASE)
    if sql_block_match:
        sql = sql_block_match.group(1).strip()
        return reasoning, sql

    # Fallback: search for SELECT statement directly
    select_match = re.search(r"\b(SELECT\b.+?;)", text_without_think, re.DOTALL | re.IGNORECASE)
    if select_match:
        sql = select_match.group(1).strip()
        return reasoning, sql

    return reasoning, None


def call_ollama(
    prompt: str,
    model: str = SLM_MODEL_NAME,
    base_url: str = OLLAMA_BASE_URL,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """
    Makes an HTTP POST request to Ollama generate endpoint.
    Zero external dependencies (uses standard library urllib.request).
    """
    endpoint = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 384,
        },
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                result = json.loads(response.read().decode("utf-8"))
                return {
                    "status": "success",
                    "response": result.get("response", ""),
                    "error": None,
                }
            return {
                "status": "error",
                "response": "",
                "error": f"Ollama returned HTTP status {response.status}",
            }
    except urllib.error.URLError as err:
        return {
            "status": "error",
            "response": "",
            "error": f"Ollama connection error ({endpoint}): {err}",
        }
    except Exception as err:
        return {
            "status": "error",
            "response": "",
            "error": f"Unexpected error communicating with Ollama: {err}",
        }


def rule_based_fallback_sql(question: str) -> Optional[str]:
    """
    Deterministic regex-based SQL generator for testing and offline resilience.
    Extracts ticker, year, report type, and accounting keyword.
    """
    q_lower = question.lower()

    # Detect Ticker
    ticker_match = re.search(r"\b([A-Z]{3})\b", question)
    ticker = ticker_match.group(1) if ticker_match else None
    if not ticker:
        if "vietjet" in q_lower:
            ticker = "VJC"
        elif "á châu" in q_lower or "acb" in q_lower:
            ticker = "ACB"
        elif "hòa phát" in q_lower or "hpg" in q_lower:
            ticker = "HPG"
        elif "vinamilk" in q_lower or "vnm" in q_lower:
            ticker = "VNM"

    if not ticker:
        return None

    # Detect Year
    year_match = re.search(r"\b(201[5-9]|202[0-5])\b", question)
    year = int(year_match.group(1)) if year_match else None

    # Detect Report Type
    report_type = "consolidated"
    if any(k in q_lower for k in ["công ty mẹ", "riêng lẻ", "me"]):
        report_type = "separate"
    elif "tổng hợp" in q_lower:
        report_type = "aggregated"

    # Detect Item Keyword
    item_keyword = None
    if "lãi tiền gửi" in q_lower:
        item_keyword = "lai tien gui"
    elif "thương mại" in q_lower:
        item_keyword = "thuong mai"
    elif "doanh thu" in q_lower:
        item_keyword = "doanh thu"
    elif "lợi nhuận" in q_lower:
        item_keyword = "loi nhuan"
    elif "vốn chủ sở hữu" in q_lower:
        item_keyword = "von chu so huu"
    elif "tiền và tương đương tiền" in q_lower:
        item_keyword = "tien va tuong duong tien"

    # Detect Unit Scaling
    unit_col = "value_vnd"
    if "triệu" in q_lower:
        unit_col = "ABS(value_vnd) / 1000000.0 AS value_trieu"
    elif "nghìn tỷ" in q_lower:
        unit_col = "ABS(value_vnd) / 1000000000000.0 AS value_nghin_ty"
    elif "tỷ" in q_lower:
        unit_col = "ABS(value_vnd) / 1000000000.0 AS value_ty"
    elif "nghìn" in q_lower:
        unit_col = "ABS(value_vnd) / 1000.0 AS value_nghin"

    conditions = [f"ticker = '{ticker}'"]
    if year:
        conditions.append(f"year = {year}")
    conditions.append(f"report_type = '{report_type}'")

    if item_keyword:
        conditions.append(f"item_name_ascii LIKE '%{item_keyword}%'")

    # Period condition
    if "đầu năm" in q_lower:
        conditions.append("(period_label LIKE '%đầu năm%' OR period_label LIKE '%01/01%' OR period_label LIKE '%1/1/%' OR period_label LIKE '%01.01%' OR period_label LIKE '%1.1.%')")
    elif "cuối năm" in q_lower:
        conditions.append("(period_label LIKE '%cuối năm%' OR period_label LIKE '%31/12%' OR period_label LIKE '%31.12%')")

    # Exclude USD
    conditions.append("(unit != 'USD' OR unit IS NULL)")

    where_clause = " AND ".join(conditions)
    sql = f"""SELECT item_name, period_label, {unit_col}, raw_value, page_no, source_doc FROM financial_facts WHERE {where_clause} ORDER BY page_no ASC LIMIT 5;"""
    return sql


def generate_sql(
    question: str,
    model: str = SLM_MODEL_NAME,
    base_url: str = OLLAMA_BASE_URL,
    allow_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Main entry point to generate SQL from user question.
    Tries Ollama first. If Ollama is offline and allow_fallback is True, uses rule-based generator.
    """
    prompt = build_prompt(question)
    ollama_res = call_ollama(prompt, model=model, base_url=base_url, timeout=120.0)

    if ollama_res["status"] == "success":
        reasoning, sql = extract_think_and_sql(ollama_res["response"])
        if sql:
            return {
                "status": "success",
                "sql_query": sql,
                "reasoning": reasoning,
                "model": model,
                "mode": "ollama_slm",
                "error": None,
            }

    # Fallback path if Ollama is unavailable or failed to produce SQL
    if allow_fallback:
        fallback_sql = rule_based_fallback_sql(question)
        if fallback_sql:
            return {
                "status": "success",
                "sql_query": fallback_sql,
                "reasoning": "Offline Rule-based Deterministic Fallback Generator",
                "model": "offline_fallback",
                "mode": "offline_rule_fallback",
                "error": ollama_res.get("error"),
            }

    return {
        "status": "error",
        "sql_query": None,
        "reasoning": None,
        "model": model,
        "mode": "failed",
        "error": ollama_res.get("error") or "Failed to extract valid SQL from response",
    }
