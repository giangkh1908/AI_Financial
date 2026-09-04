# KIẾN TRÚC TẦNG SUY LUẬN (INFERENCE & SERVING SPECIFICATION)

> **Trạng thái tài liệu**: `[THIẾT KẾ MỤC TIÊU / SPECIFICATION ROADMAP]`  
> Tài liệu này đặc tả kiến trúc kỹ thuật của tầng suy luận (Inference Layer) phục vụ triển khai Chatbot API. Tầng Dữ liệu & ETL bên dưới đã hoàn thành thực tế tại `scripts/build_db.py` và `data/financial.db`.

---

## 1. TỔNG QUAN LUỒNG SUY LUẬN (STRUCTURED-FIRST HYBRID RAG)

Hệ thống hoạt động theo triết lý **Structured-First Hybrid RAG**: 90% câu hỏi số liệu tài chính được giải quyết bằng Text-to-SQL thực thi trực tiếp trên CSDL quan hệ (đảm bảo độ chính xác số học 100% và Zero Hallucination cho số liệu), 10% câu hỏi định tính được chuyển sang bộ đọc văn bản bán cấu trúc (Document Reader kết hợp FTS5).

```
                          [Câu hỏi của người dùng]
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Intent & Entity Router                                                   │
│    • Nhận diện thực thể: Ticker, Năm, Report Scope, Kỳ kế toán              │
│    • Phân luồng: Định lượng (Số học) vs Định tính (Chính sách/Ý kiến)       │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼ (90% Câu hỏi Số liệu)                 ▼ (10% Câu hỏi Văn bản)
┌────────────────────────────────────────┐ ┌────────────────────────────────────────┐
│ 2A. Text-to-SQL Engine                 │ │ 2B. Qualitative Document Reader        │
│ • Model: Qwen3.5-4B-SQL Local          │ │ • Đọc thuyết minh (statement='notes')  │
│ • Native Chain-of-Thought (<think>)    │ │   kết hợp truy xuất text file theo     │
│ • Schema 15 cột chuẩn hóa              │ │   page_no và source_doc                │
└──────────────────┬─────────────────────┘ └───────────────────┬────────────────────┘
                   │                                           │
                   ▼                                           ▼
┌────────────────────────────────────────┐                     │
│ 3. SQL Safety Guard & Sandbox Execution│                     │
│ • Chế độ Read-Only tuyệt đối           │                     │
│ • Chống Full-Table Scan & Double Count │                     │
│ • Cơ chế FTS5 Fallback khi rỗng        │                     │
└──────────────────┬─────────────────────┘                     │
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. Grounding & Response Formatter                                           │
│    • Định dạng số tiền (Tỷ / Triệu VNĐ) và tỷ lệ %                          │
│    • Đính kèm minh chứng kiểm toán: Mã CK, Tài liệu, Trang BCTC             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. CÁC THÀNH PHẦN CỐT LÕI (CORE COMPONENTS)

### 2.1. Intent & Entity Router (Bộ điều phối ý định & thực thể)

Hoạt động theo cơ chế **2-Stage Hybrid Router**:

#### Stage 1: Nhận diện Thực thể Cứng (Deterministic Regex & Lookup - < 1ms)
* **Mã chứng khoán (Ticker)**: Tra cứu nhanh từ bảng danh mục 100 mã (`data/code_stock.csv`) hoặc tên đầy đủ (ví dụ: *"Vietjet"* $\rightarrow$ `VJC`, *"Á Châu"* $\rightarrow$ `ACB`, *"Hòa Phát"* $\rightarrow$ `HPG`).
* **Năm tài chính (Year)**: Regex `\b(201[5-9]|202[0-5])\b`.
* **Phạm vi báo cáo (report_type)**:
  * Chứa từ khóa *"công ty mẹ"*, *"riêng lẻ"* $\rightarrow$ `report_type = 'separate'`
  * Chứa từ khóa *"tổng hợp"* $\rightarrow$ `report_type = 'aggregated'` (Áp dụng cho ACV, DTK, SJG... có 7.833 dòng thực tế)
  * Mặc định hoặc chứa *"hợp nhất"* $\rightarrow$ `report_type = 'consolidated'`
* **Kỳ kế toán (Period Mapping)**:
  * Chứa *"đầu năm"* $\rightarrow$ ánh xạ `period_label LIKE '%đầu năm%'` hoặc `LIKE '%01/01%'`
  * Chứa *"cuối năm"* $\rightarrow$ ánh xạ `period_label LIKE '%cuối năm%'` hoặc `LIKE '%31/12%'`

#### Stage 2: Phân luồng Ý định (Branch Classification)
* **Nhánh Số liệu (Quantitative)**: Câu hỏi chứa từ khóa số lượng (`lãi`, `doanh thu`, `chi phí`, `nợ`, `tài sản`, `cho vay`, `vốn`, `bao nhiêu`, `tăng trưởng`, `so sánh`) $\rightarrow$ Chuyển sang **Text-to-SQL Engine**.
* **Nhánh Định tính (Qualitative)**: Câu hỏi hỏi về văn bản (`ý kiến kiểm toán`, `chấp nhận toàn phần`, `ngoại trừ`, `nguyên tắc ghi nhận`, `chính sách kế toán`) $\rightarrow$ Chuyển sang **Qualitative Document Reader**.

---

### 2.2. Text-to-SQL Generation Engine

#### 1. Xuất xứ & Cấu hình Mô hình (Model Lineage)
* **Base Model**: `Qwen/Qwen3.5-4B`
* **Fine-tuned Adapter**: [`giangkh19/qwen3.5-4b-sql`](https://huggingface.co/giangkh19/qwen3.5-4b-sql) (Huấn luyện QLoRA $r=16, \alpha=32$ trên 17.000 mẫu truy vấn tài chính đa bảng phức tạp có sinh thẻ suy luận nội tại `<think> ... </think>`).
* **Serving Model**: [`giangkh19/qwen3.5-4b-sql-gguf:Q4_K_M`](https://huggingface.co/giangkh19/qwen3.5-4b-sql-gguf) (Chạy cục bộ qua Ollama hoặc llama.cpp, RAM ~3GB, độ trễ <200ms).

#### 2. Schema Prompting chuẩn hóa (Đồng bộ 15 cột thực tế)
Prompt cung cấp cho model cấu trúc chuẩn xác của bảng `financial_facts`:
```text
Table: financial_facts
Columns:
- ticker: TEXT (Mã cổ phiếu: 'VJC', 'ACB', 'HPG', 'ACV')
- company_name: TEXT (Tên đầy đủ: 'CTCP Hàng không Vietjet')
- year: INTEGER (Năm báo cáo: 2015 đến 2025)
- report_type: TEXT ('consolidated' [Hợp nhất], 'separate' [Mẹ/Riêng lẻ], 'aggregated' [Tổng hợp])
- statement: TEXT ('balance_sheet', 'income_statement', 'cash_flow', 'notes')
- section_title: TEXT (Tên phân mục / Thuyết minh: 'Thuyết minh số 28 - Doanh thu tài chính')
- item_code: TEXT (Mã số kế toán: '100', '110', '05', 'I', 'II')
- item_name: TEXT (Tên chỉ tiêu có dấu gốc)
- item_name_ascii: TEXT (Tên chỉ tiêu viết thường không dấu: 'lai tien gui va cho vay')
- period_label: TEXT (Kỳ báo cáo: 'Số cuối năm', '31/12/2022', '2018VND')
- value_vnd: REAL (Giá trị số thực quy chuẩn về VNĐ)
- raw_value: TEXT (Chuỗi gốc trong bảng OCR: '(208.253.201.298)')
- unit: TEXT (Đơn vị tính gốc: 'VND', 'Triệu VND', 'USD')
- page_no: INTEGER (Số trang chính xác trong tài liệu BCTC)
- source_doc: TEXT (Tên tài liệu: 'VJC_financial_statements_2018_separate')
```

#### 3. Bảng Ánh xạ Quy đổi Đơn vị & Hệ số Tiền tệ (Unit Conversion Policy)
Khảo sát trên 1.012 câu hỏi thực tế trong `questions.jsonl`:
| Đơn vị trong câu hỏi | Tần suất xuất hiện | Quy tắc tính toán trong SQL |
| :--- | :---: | :--- |
| **Nghìn tỷ đồng** | 85 câu | `ABS(value_vnd) / 1000000000000.0 AS value_nghin_ty` |
| **Trăm tỷ đồng** | 66 câu | `ABS(value_vnd) / 100000000000.0 AS value_tram_ty` |
| **Tỷ đồng** | Phổ biến | `ABS(value_vnd) / 1000000000.0 AS value_ty` |
| **Triệu đồng** | Rất phổ biến | `ABS(value_vnd) / 1000000.0 AS value_trieu` |
| **Nghìn đồng** | 16 câu | `ABS(value_vnd) / 1000.0 AS value_nghin` |
| **Tỷ lệ phần trăm (%)** | 260 câu | Trích xuất `raw_value` hoặc tính tỷ lệ giữa 2 chỉ tiêu, **không chia hệ số tiền tệ**. |
| **Cổ phiếu / Cổ phần** | 144 câu | Trích xuất `raw_value` (ví dụ: số lượng CP đang lưu hành, EPS). |

#### 4. Chính sách xử lý dữ liệu USD (USD Exclusion Policy)
Trong CSDL có **30.556 bản ghi** mang `unit = 'USD'` (factor lưu trữ = 1.0).
* Nếu câu hỏi bằng VNĐ: Bắt buộc thêm điều kiện `AND (unit != 'USD' OR unit IS NULL)` để tránh chia nhầm `1e6` trên số USD.
* Nếu câu hỏi hỏi rõ số USD (ví dụ: Q191, Q213): Thêm điều kiện `AND unit = 'USD'` và trả về `raw_value` hoặc `value_vnd` kèm đơn vị USD.

---

### 2.3. SQL Safety Guard & Quy tắc chống Anti-Patterns

Để bảo vệ CSDL và ngăn ngừa sai lệch dữ liệu:
1. **Chế độ Read-Only tuyệt đối**: Mọi kết nối CSDL từ tầng Inference đều mở ở chế độ `file:data/financial.db?mode=ro`. Chặn đứng mọi câu lệnh `DROP`, `INSERT`, `UPDATE`, `DELETE`, `ALTER`.
2. **Chống quét Full Table Scan trên 2,1 triệu dòng**:
   * Cú pháp `LIKE '%từ_khóa%'` (có wildcard đầu) sẽ bỏ qua B-Tree index.
   * **Quy chuẩn bắt buộc**: Sử dụng bảng ảo `facts_fts` (`WHERE id IN (SELECT rowid FROM facts_fts WHERE facts_fts MATCH 'từ_khóa')`) hoặc dùng prefix matching `item_name_ascii LIKE 'từ_khóa%'` kết hợp lọc `ticker`, `year`, `report_type` trước.
3. **NGHIÊM CẤM DÙNG `SUM()` TRÊN KẾT QUẢ SO KHỚP CHUỖI MỜ (Anti-Double-Counting Rule)**:
   * Nếu chạy `SUM(value_vnd) WHERE item_name_ascii LIKE '%tai san%'`, CSDL sẽ cộng dồn cả dòng Tổng (Mã 100: Tài sản ngắn hạn) lẫn các dòng con (Mã 110: Tiền, Mã 120: Đầu tư), dẫn đến kết quả sai gấp đôi.
   * `SUM()` chỉ được phép dùng khi nhóm theo phân loại danh mục hoặc đã chỉ định rõ các `item_code` độc lập.
4. **Quy tắc Giới hạn dòng (LIMIT Rule)**:
   * Áp dụng `LIMIT 10` cho các câu truy vấn tra cứu danh sách bản ghi.
   * **Không áp dụng LIMIT** đối với các câu lệnh tính toán gộp vô hướng (`SELECT SUM(...)`, `SELECT COUNT(...)`).

---

### 2.4. Qualitative Document Reader (Nhánh đọc tài liệu định tính)

Đối với 10% các câu hỏi định tính (ý kiến kiểm toán, chính sách khấu hao):
* Dữ liệu định tính được truy xuất từ 2 nguồn:
  1. Các bản ghi trong `financial_facts` có `statement = 'notes'` (chiếm 81,9% dữ liệu - 1.733.823 dòng).
  2. Đọc trực tiếp các đoạn văn bản gốc từ file `_extracted.txt` theo cặp khóa `(source_doc, page_no)` đã được đánh chỉ mục.
* Trích xuất đoạn văn liên quan và đưa vào Context của LLM để tóm tắt cho người dùng.

---

### 2.5. Grounding & Response Formatter (Tầng định dạng câu trả lời)

* **Phạm vi cam kết Zero Hallucination**:
  * Cam kết **0.0% ảo giác đối với các con số tài chính trích xuất từ CSDL thông qua câu lệnh SQL**.
  * Các câu diễn giải văn phong tự nhiên được bám sát theo nguyên văn báo cáo kiểm toán.
* **Quy chuẩn dẫn chứng kiểm toán (Audit Provenance)**:
  Mỗi câu trả lời số liệu tài chính bắt buộc đính kèm khối trích dẫn:
  ```markdown
  📌 Nguồn dữ liệu kiểm chứng:
  • Doanh nghiệp: CTCP Hàng không Vietjet (VJC)
  • Báo cáo: Báo cáo tài chính kiểm toán công ty mẹ năm 2018 (VJC_financial_statements_2018_separate)
  • Vị trí: Trang 10 (Lưu chuyển tiền tệ) và Trang 44 (Thuyết minh 28 - Doanh thu tài chính)
  • Giá trị gốc trong bảng: 208.253.201.298 VNĐ
  ```

---

## 3. THIẾT KẾ REST API CONTRACT (SERVING INTERFACE)

Hệ thống cung cấp giao diện REST API chuẩn hóa (FastAPI):

### 3.1. Endpoint `POST /api/v1/chat/query`
Thực hiện truy vấn câu hỏi tài chính từ người dùng.

#### Request Schema
```json
{
  "question": "Lãi tiền gửi năm 2018 của công ty mẹ Vietjet (VJC) là bao nhiêu triệu đồng?",
  "session_id": "optional-session-uuid",
  "temperature": 0.0,
  "model_preference": "local_slm"
}
```

#### Response Schema
```json
{
  "status": "success",
  "question": "Lãi tiền gửi năm 2018 của công ty mẹ Vietjet (VJC) là bao nhiêu triệu đồng?",
  "answer": "Lãi tiền gửi năm 2018 của công ty mẹ Vietjet (VJC) là 208.253,2 triệu đồng.",
  "sql_query": "SELECT item_name, ABS(value_vnd) / 1000000.0 AS value_trieu, raw_value, page_no, source_doc FROM financial_facts WHERE ticker = 'VJC' AND year = 2018 AND report_type = 'separate' AND item_name_ascii LIKE '%lai tien gui%' LIMIT 5;",
  "execution_time_ms": 142.5,
  "provenance": [
    {
      "ticker": "VJC",
      "company_name": "CTCP Hàng không Vietjet",
      "source_doc": "VJC_financial_statements_2018_separate",
      "page_no": 10,
      "statement": "cash_flow",
      "raw_value": "(208.253.201.298)"
    },
    {
      "ticker": "VJC",
      "company_name": "CTCP Hàng không Vietjet",
      "source_doc": "VJC_financial_statements_2018_separate",
      "page_no": 44,
      "statement": "notes",
      "raw_value": "208.253.201.298"
    }
  ]
}
```

### 3.2. Endpoint `GET /api/v1/health`
Kiểm tra tính khả dụng của hệ thống (CSDL SQLite, Model Inference Engine).
```json
{
  "status": "healthy",
  "database_connected": true,
  "database_records": 2116243,
  "model_loaded": "giangkh19/qwen3.5-4b-sql-gguf:Q4_K_M"
}
```

---

## 4. MA TRẬN ĐÁNH GIÁ (EVALUATION ROADMAP)

Để đánh giá chất lượng toàn diện:
1. **Annotated Gold Benchmark**: Cần xây dựng tập đáp án chuẩn (Gold Answer Dataset) chọn lọc từ 1.012 câu hỏi của `questions.jsonl` làm thước đo độc lập.
2. **Execution Accuracy (EX-Acc)**: Đo lường tỷ lệ câu SQL sinh ra cho kết quả khớp chính xác với đáp án chuẩn.
3. **Provenance Recall**: Tỷ lệ truy xuất trúng số trang (`page_no`) và tài liệu gốc (`source_doc`).
4. **Latency Benchmark**: Thời gian phản hồi trung bình < 500ms đối với truy vấn số liệu cục bộ.
