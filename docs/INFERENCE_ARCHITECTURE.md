# KIẾN TRÚC SUY LUẬN & PHÂN TÍCH TÀI CHÍNH (TEXT-TO-SQL & QUANTITATIVE ANALYTICS ENGINE)

> **Trạng thái tài liệu**: `[THIẾT KẾ MỤC TIÊU / TARGET SPECIFICATION]`  
> Đặc tả kiến trúc phục vụ tầng suy luận (Inference Layer) và phân tích định lượng cho Chatbot API. Hệ thống tập trung **100% vào dữ liệu tài chính có cấu trúc** từ CSDL SQLite (`data/financial.db` gồm 2.116.243 facts qua 10 năm 2015–2025 của 100 doanh nghiệp niêm yết).

---

## 1. TỔNG QUAN KIẾN TRÚC HỆ THỐNG

Hệ thống hoạt động theo triết lý **Deterministic Text-to-SQL & Quantitative Analytics**:
* **100% Dữ liệu có cấu trúc**: Loại bỏ hoàn toàn vector search / document reader định tính rườm rà. Toàn bộ 2,1 triệu facts (bao gồm cả Bảng cân đối, Kết quả kinh doanh, Lưu chuyển tiền tệ và 1.733.823 dòng Thuyết minh BCTC) được truy vấn thông qua SQL chuẩn xác trên SQLite Native (B-Tree + FTS5).
* **Zero Hallucination cho số liệu**: Không để LLM tự làm toán hay đoán số. LLM chỉ đảm nhận vai trò dịch ngôn ngữ tự nhiên thành SQL; khâu tính toán số học, tổng hợp, và phân tích đều do SQLite và Python Engine thực thi.
* **Tích hợp Module Chẩn đoán & Dự báo**: Tận dụng chuỗi dữ liệu lịch sử 10 năm để tính toán các mô hình kinh điển (Altman Z-Score, Piotroski F-Score, DuPont) và dự báo chuỗi thời gian (Time-Series Projections).

```
                          [Câu hỏi của người dùng]
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Entity Normalizer & Intent Preprocessor (< 1ms)                          │
│    • Bóc tách Ticker (100 mã), Năm (2015-2025), Loại BCTC, Kỳ kế toán       │
│    • Định tuyến: Truy vấn Dữ liệu (Fact Query) vs Phân tích/Dự báo (Analytics)│
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. Text-to-SQL Generation Engine                                            │
│    • SLM: Qwen3.5-4B-SQL Local (GGUF Q4_K_M qua Ollama / llama.cpp)         │
│    • Sinh chuỗi suy luận nội tại (<think> ... </think>)                     │
│    • Prompt schema 15 cột chuẩn hóa và quy tắc loại trừ USD                 │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. SQL Safety Guard & Sandbox Execution (SQLite Native - < 2ms)             │
│    • Chế độ Read-Only tuyệt đối (file:financial.db?mode=ro)                 │
│    • Kiểm tra Anti-Pattern: Chặn Full-Table Scan, Chặn SUM() mù             │
│    • Tự động fallback FTS5 Virtual Table khi chuỗi tìm kiếm biến thể        │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. Financial Analytics & Forecasting Engine (Phân tích & Dự báo Định lượng) │
│    • Chẩn đoán Sức khỏe Doanh nghiệp: Altman Z-Score, Piotroski F-Score     │
│    • Phân rã Hiệu quả Hoạt động: Mô hình DuPont 3 bước & 5 bước             │
│    • Dự báo Chuỗi thời gian: CAGR 3-5-10 năm, Dự phóng Doanh thu/Lợi nhuận  │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. Grounding & Response Formatter                                           │
│    • Định dạng số học tài chính chuẩn mực (Tỷ / Triệu VNĐ, %)               │
│    • Trích dẫn kiểm toán Provenance: Mã CK, Báo cáo, Trang BCTC, Số gốc     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. CÁC THÀNH PHẦN CỐT LÕI (CORE COMPONENTS)

### 2.1. Entity Normalizer & Intent Preprocessor (Bộ tiền xử lý thực thể)

Tiền xử lý chuỗi truy vấn bằng thuật toán tất định (Deterministic Lookup & Regex - độ trễ < 1ms):

1. **Mã chứng khoán (Ticker)**: Tra cứu nhanh từ bảng danh mục 100 mã (`data/code_stock.csv`) hoặc tên thường gọi (ví dụ: *"Vietjet"* $\rightarrow$ `VJC`, *"Á Châu"* $\rightarrow$ `ACB`, *"Hòa Phát"* $\rightarrow$ `HPG`, *"Vinamilk"* $\rightarrow$ `VNM`).
2. **Năm tài chính (Year)**: Regex `\b(201[5-9]|202[0-5])\b`.
3. **Phạm vi báo cáo (report_type)**:
   * Chứa từ khóa *"công ty mẹ"*, *"riêng lẻ"* $\rightarrow$ `report_type = 'separate'`
   * Chứa từ khóa *"tổng hợp"* $\rightarrow$ `report_type = 'aggregated'` (Áp dụng cho ACV, DTK, SJG... có 7.833 dòng thực tế)
   * Mặc định hoặc chứa *"hợp nhất"* $\rightarrow$ `report_type = 'consolidated'`
4. **Kỳ kế toán (Period Mapping)**:
   * Chứa *"đầu năm"* $\rightarrow$ ánh xạ:
     `(period_label LIKE '%đầu năm%' OR period_label LIKE '%01/01%' OR period_label LIKE '%1/1/%' OR period_label LIKE '%01.01%' OR period_label LIKE '%1.1.%')`
   * Chứa *"cuối năm"* $\rightarrow$ ánh xạ:
     `(period_label LIKE '%cuối năm%' OR period_label LIKE '%31/12%' OR period_label LIKE '%31.12%')`
5. **Phân loại Ý định (Intent Classification)**:
   * **Tra cứu Số liệu (Fact Lookup)**: Các câu hỏi điểm số cụ thể (Doanh thu năm 2023, nợ vay, lãi tiền gửi...).
   * **Phân tích & Dự báo (Analytics & Forecast)**: Các câu hỏi chứa từ khóa: *"sức khỏe tài chính"*, *"nguy cơ phá sản"*, *"Z-Score"*, *"F-Score"*, *"DuPont"*, *"dự báo"*, *"tăng trưởng năm tới"*, *"xu hướng"*.

---

### 2.2. Text-to-SQL Generation Engine

#### 1. Xuất xứ & Cấu hình Mô hình (Model Lineage)
* **Base Model**: `Qwen/Qwen3.5-4B`
* **Fine-tuned Adapter**: [`giangkh19/qwen3.5-4b-sql`](https://huggingface.co/giangkh19/qwen3.5-4b-sql) (Huấn luyện QLoRA $r=16, \alpha=32$ trên 17.000 mẫu truy vấn tài chính đa bảng phức tạp có sinh thẻ suy luận nội tại `<think> ... </think>`).
* **Serving Model**: [`giangkh19/qwen3.5-4b-sql-gguf:Q4_K_M`](https://huggingface.co/giangkh19/qwen3.5-4b-sql-gguf) (Chạy cục bộ qua Ollama hoặc llama.cpp, RAM ~3GB, độ trễ <200ms).

#### 2. Schema Prompting chuẩn hóa (15 cột nghiệp vụ)
```text
Table: financial_facts
Columns:
- ticker: TEXT (Mã cổ phiếu: 'VJC', 'ACB', 'HPG', 'ACV')
- company_name: TEXT (Tên đầy đủ: 'CTCP Hàng không Vietjet')
- year: INTEGER (Năm báo cáo: 2015 đến 2025)
- report_type: TEXT ('consolidated' [Hợp nhất], 'separate' [Mẹ/Riêng lẻ], 'aggregated' [Tổng hợp])
- statement: TEXT ('balance_sheet', 'income_statement', 'cash_flow', 'notes')
- section_title: TEXT (Tên phân mục / Thuyết minh: '29. Doanh thu hoạt động tài chính')
- item_code: TEXT (Mã số kế toán: '100', '110', '05', 'I', 'II')
- item_name: TEXT (Tên chỉ tiêu có dấu gốc)
- item_name_ascii: TEXT (Tên chỉ tiêu viết thường không dấu: 'lai tien gui va cho vay')
- period_label: TEXT (Kỳ báo cáo: 'Số cuối năm', '31/12/2022', '31.12.2022Triệu VND')
- value_vnd: REAL (Giá trị số thực quy chuẩn về VNĐ)
- raw_value: TEXT (Chuỗi gốc trong bảng OCR: '(208.253.201.298)')
- unit: TEXT (Đơn vị tính gốc: 'VND', 'Triệu VND', 'USD')
- page_no: INTEGER (Số trang chính xác trong tài liệu BCTC)
- source_doc: TEXT (Tên tài liệu: 'VJC_financial_statements_2018_separate')
```

#### 3. Quy tắc Đơn vị & Hệ số Tiền tệ (Unit Conversion Policy)
Khảo sát trên 1.012 câu hỏi thực tế trong `data/questions/questions.jsonl`:
| Đơn vị trong câu hỏi | Tần suất xuất hiện | Quy tắc tính toán trong SQL |
| :--- | :---: | :--- |
| **Nghìn tỷ đồng** | 85 câu | `ABS(value_vnd) / 1000000000000.0 AS value_nghin_ty` |
| **Trăm tỷ đồng** | 66 câu | `ABS(value_vnd) / 100000000000.0 AS value_tram_ty` |
| **Tỷ đồng** | Phổ biến (391 câu) | `ABS(value_vnd) / 1000000000.0 AS value_ty` |
| **Triệu đồng** | Rất phổ biến (218 câu) | `ABS(value_vnd) / 1000000.0 AS value_trieu` |
| **Nghìn đồng** | 17 câu (16 nghìn + 1 ngàn) | `ABS(value_vnd) / 1000.0 AS value_nghin` |
| **Tỷ lệ phần trăm (%)** | 260 câu | Trích xuất `raw_value` hoặc tính tỷ lệ giữa 2 chỉ tiêu, **không chia hệ số tiền tệ**. |
| **Cổ phiếu / Cổ phần** | 107 câu unique (124 lượt) | Trích xuất `raw_value` (ví dụ: số lượng CP đang lưu hành, EPS). |

#### 4. Chính sách xử lý dữ liệu USD (USD Exclusion Policy)
Trong CSDL có **30.556 bản ghi** mang `unit = 'USD'` (factor lưu trữ = 1.0).
* Nếu câu hỏi bằng VNĐ: Bắt buộc thêm điều kiện `AND (unit != 'USD' OR unit IS NULL)` để tránh chia nhầm `1e6` trên số USD.
* Nếu câu hỏi hỏi rõ số USD (ví dụ: Q191, Q213): Thêm điều kiện `AND unit = 'USD'` và trả về `raw_value` hoặc `value_vnd` kèm đơn vị USD.

---

### 2.3. SQL Safety Guard & Chống Anti-Patterns

1. **Chế độ Read-Only tuyệt đối**: Mọi kết nối CSDL từ tầng Inference đều mở ở chế độ `file:data/financial.db?mode=ro`. Chặn đứng mọi câu lệnh `DROP`, `INSERT`, `UPDATE`, `DELETE`, `ALTER`.
2. **Chống quét Full Table Scan trên 2,1 triệu dòng**:
   * Cú pháp `LIKE '%từ_khóa%'` (wildcard đầu) sẽ bỏ qua B-Tree index.
   * **Quy chuẩn bắt buộc**: Sử dụng bảng ảo `facts_fts` (`WHERE id IN (SELECT rowid FROM facts_fts WHERE facts_fts MATCH 'từ_khóa')`) hoặc dùng prefix matching `item_name_ascii LIKE 'từ_khóa%'` kết hợp lọc `ticker`, `year`, `report_type` trước.
3. **NGHIÊM CẤM DÙNG `SUM()` TRÊN KẾT QUẢ SO KHỚP CHUỖI MỜ (Anti-Double-Counting Rule)**:
   * Nếu chạy `SUM(value_vnd) WHERE item_name_ascii LIKE '%tai san%'`, CSDL sẽ cộng dồn cả dòng Tổng (Mã 100: Tài sản ngắn hạn) lẫn các dòng con (Mã 110, 120...), dẫn đến kết quả sai gấp đôi.
   * `SUM()` chỉ được phép dùng khi đã chỉ định rõ các `item_code` độc lập hoặc nhóm theo danh mục.
4. **Quy tắc Giới hạn dòng (LIMIT Rule)**: Áp dụng `LIMIT 10` cho các câu truy vấn danh sách bản ghi; không áp dụng LIMIT đối với các câu lệnh tính toán vô hướng (`SELECT COUNT(*)`, `SELECT AVG(...)`).

---

### 2.4. Module Phân tích & Dự báo Tài chính (Quantitative Analytics & Forecasting)

Đây là tầng nâng cấp chiến lược, biến trợ lý từ tra cứu số liệu thụ động thành **Công cụ Phân tích Tài chính Chủ động**:

```
[SQL Engine trích xuất Time-Series Facts 2015-2025]
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Diagnostic Health Scoring (Chẩn đoán Sức khỏe)                           │
│    • Altman Z-Score: Dự báo nguy cơ kiệt quệ tài chính trong 2 năm tới      │
│    • Piotroski F-Score: Thang điểm 0-9 đánh giá cải thiện sức mạnh cơ bản   │
│    • Beneish M-Score: Phát hiện dấu hiệu gian lận / thao túng lợi nhuận     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. Performance Decomposition (Phân rã Hiệu quả)                             │
│    • DuPont 3-Step: ROE = Net Margin × Asset Turnover × Equity Multiplier   │
│    • Phân tích chu kỳ tiền mặt (Cash Conversion Cycle - CCC)                │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. Time-Series Trend & Projections (Dự báo Chuỗi thời gian)                 │
│    • CAGR 3 năm, 5 năm, 10 năm cho Doanh thu và Lợi nhuận                   │
│    • Hồi quy xu hướng tuyến tính (Linear Trend) + Khoảng tin cậy 95%        │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 1. Mô hình Chẩn đoán Nguy cơ Phá sản (Altman Z-Score)
Áp dụng cho doanh nghiệp sản xuất niêm yết:
$$Z = 1.2 X_1 + 1.4 X_2 + 3.3 X_3 + 0.6 X_4 + 0.999 X_5$$
Trong đó:
* $X_1 = \text{Vốn lưu động ròng} / \text{Tổng tài sản}$
* $X_2 = \text{Lợi nhuận sau thuế chưa phân phối} / \text{Tổng tài sản}$
* $X_3 = \text{Lợi nhuận trước thuế và lãi vay (EBIT)} / \text{Tổng tài sản}$
* $X_4 = \text{Vốn hóa thị trường (hoặc Vốn chủ sở hữu)} / \text{Tổng nợ phải trả}$
* $X_5 = \text{Doanh thu thuần} / \text{Tổng tài sản}$

**Ngưỡng phân định (Zones of Discrimination)**:
* $Z > 2.99$: **Vùng An toàn (Safe Zone)** — Tình hình tài chính vững chắc.
* $1.81 \le Z \le 2.99$: **Vùng Cảnh báo (Grey Zone)** — Cần thận trọng theo dõi.
* $Z < 1.81$: **Vùng Nguy hiểm (Distress Zone)** — Nguy cơ kiệt quệ tài chính cao trong 2 năm tới.

#### 2. Thang điểm Đánh giá Sức mạnh Cơ bản (Piotroski F-Score)
Thang điểm 9 tiêu chí nhị phân ($0 \text{ hoặc } 1$ điểm):
* **Nhóm Khả năng sinh lời**:
  1. $ROA > 0$ (+1)
  2. Dòng tiền từ hoạt động kinh doanh $CFO > 0$ (+1)
  3. $\Delta ROA > 0$ (ROA năm nay cao hơn năm trước) (+1)
  4. $CFO > \text{Lợi nhuận sau thuế}$ (Chất lượng lợi nhuận cao) (+1)
* **Nhóm Đòn bẩy & Khả năng thanh toán**:
  5. Tỷ lệ đòn bẩy nợ dài hạn giảm (+1)
  6. Tỷ số thanh toán hiện hành tăng (+1)
  7. Không phát hành cổ phiếu mới làm pha loãng (+1)
* **Nhóm Hiệu quả vận hành**:
  8. Biên lợi nhuận gộp tăng (+1)
  9. Vòng quay tài sản tăng (+1)

**Đánh giá**:
* **8 - 9 điểm**: Doanh nghiệp cực kỳ vững mạnh, sức khỏe tài chính cải thiện vượt bậc.
* **0 - 2 điểm**: Doanh nghiệp có dấu hiệu suy thoái nghiêm trọng về cơ bản.

#### 3. Phân rã Lợi nhuận DuPont 3 bước
$$ROE = \frac{\text{Lợi nhuận sau thuế}}{\text{Doanh thu thuần}} \times \frac{\text{Doanh thu thuần}}{\text{Tổng tài sản bình quân}} \times \frac{\text{Tổng tài sản bình quân}}{\text{Vốn chủ sở hữu bình quân}}$$
Giúp trả lời rõ ràng: Lợi nhuận tăng trưởng là do **tối ưu giá vốn** (biên lãi), do **khai thác tốt tài sản** (vòng quay), hay do **lạm dụng đòn bẩy nợ**.

#### 4. Dự báo Chuỗi thời gian (Time-Series Projections - Pure Python Implementation)
* **Triết lý Ponytail & Zero-External-Deps**: Toàn bộ thuật toán được tự xây dựng bằng **Standard Library Python (`math`, `statistics`)**, hoàn toàn không phụ thuộc vào `numpy`, `scipy` hay `statsmodels`:
  * Độ dốc $\beta = \frac{\sum (x - \bar{x})(y - \bar{y})}{\sum (x - \bar{x})^2}$
  * Hệ số chặn $\alpha = \bar{y} - \beta \bar{x}$
  * Sai số chuẩn $SE = \sqrt{\frac{\sum (y - \hat{y})^2}{n - 2}}$
* Tính toán **CAGR (Tốc độ tăng trưởng kép)** chuỗi 3 năm, 5 năm, 10 năm:
  $$CAGR = \left( \frac{V_{\text{cuối}}}{V_{\text{đầu}}} \right)^{\frac{1}{N}} - 1$$
* **Dự báo Doanh thu / Lợi nhuận năm tới ($T+1$)**:
  * Chạy hồi quy xu hướng tuyến tính (Ordinary Least Squares - OLS) trên chuỗi dữ liệu lịch sử sạch.
  * Cung cấp dự báo kèm **khoảng tin cậy 95%** ($\hat{y} \pm 1.96 \cdot SE$).

---

### 2.5. Grounding & Response Formatter (Tầng định dạng câu trả lời)

* **Cam kết Zero Hallucination**:
  * 100% con số tài chính trích xuất từ CSDL thông qua SQL hoặc tính toán bằng công thức toán học xác định.
  * Tuyệt đối không sinh số ước lượng từ tri thức ẩn của LLM.
* **Quy chuẩn trích dẫn kiểm toán (Audit Provenance)**:
  Mỗi câu trả lời số liệu tài chính bắt buộc đính kèm khối trích dẫn:
  ```markdown
  📌 Nguồn dữ liệu kiểm chứng:
  • Doanh nghiệp: CTCP Hàng không Vietjet (VJC)
  • Báo cáo: Báo cáo tài chính kiểm toán công ty mẹ năm 2018 (VJC_financial_statements_2018_separate)
  • Vị trí: Trang 10 (Lưu chuyển tiền tệ) và Trang 44 (Thuyết minh 29 - Doanh thu hoạt động tài chính)
  • Giá trị gốc trong bảng: 208.253.201.298 VNĐ
  ```

---

## 3. THIẾT KẾ REST API CONTRACT (SERVING INTERFACE)

Hệ thống cung cấp giao diện REST API chuẩn hóa (FastAPI):

### 3.1. Endpoint `POST /api/v1/chat/query`
Truy vấn câu hỏi tài chính tự nhiên.
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

### 3.2. Endpoint `POST /api/v1/analytics/diagnostics`
Phân tích sức khỏe tài chính tự động (Z-Score, F-Score, DuPont).
#### Request Schema
```json
{
  "ticker": "HPG",
  "year": 2023,
  "report_type": "consolidated"
}
```

#### Response Schema
```json
{
  "status": "success",
  "ticker": "HPG",
  "year": 2023,
  "altman_z_score": {
    "score": 2.85,
    "zone": "Grey Zone (Tiệm cận Safe Zone)",
    "components": { "x1": 0.18, "x2": 0.32, "x3": 0.10, "x4": 1.25, "x5": 0.78 }
  },
  "piotroski_f_score": {
    "score": 7,
    "max_score": 9,
    "rating": "Strong Financial Health",
    "signals": ["ROA_positive", "CFO_positive", "CFO_greater_than_NI", "margin_expansion"]
  },
  "dupont_analysis": {
    "roe": 0.071,
    "net_profit_margin": 0.057,
    "asset_turnover": 0.642,
    "equity_multiplier": 1.942
  },
  "execution_time_ms": 18.2
}
```

### 3.3. Endpoint `POST /api/v1/analytics/forecast`
Dự báo chuỗi thời gian cho doanh thu và lợi nhuận.
#### Request Schema
```json
{
  "ticker": "HPG",
  "target_indicator": "revenue",
  "history_years": 8,
  "forecast_horizon": 2
}
```

#### Response Schema
```json
{
  "status": "success",
  "ticker": "HPG",
  "historical_cagr": 0.142,
  "projections": [
    {
      "year": 2024,
      "predicted_value_vnd": 142500000000000.0,
      "predicted_formatted": "142.500 tỷ VNĐ",
      "confidence_interval_95": {
        "lower_bound_vnd": 135000000000000.0,
        "upper_bound_vnd": 150000000000000.0
      }
    }
  ],
  "methodology": "Linear Trend Extrapolation with 95% Confidence Interval"
}
```

### 3.4. Endpoint `GET /api/v1/health`
Kiểm tra tính khả dụng của hệ thống:
```json
{
  "status": "healthy",
  "database_connected": true,
  "database_records": 2116243,
  "model_loaded": "giangkh19/qwen3.5-4b-sql-gguf:Q4_K_M",
  "analytics_engine": "ready"
}
```

---

## 4. MA TRẬN ĐÁNH GIÁ (EVALUATION ROADMAP)

Hệ thống được đánh giá qua 4 tiêu chuẩn định lượng nghiêm ngặt:
1. **Execution Accuracy (EX-Acc)**: Đo lường tỷ lệ câu SQL sinh ra cho kết quả số học khớp 100% với đáp án chuẩn trong benchmark `data/questions/questions.jsonl` (Mục tiêu: > 85%).
2. **Provenance Recall**: Tỷ lệ câu trả lời trích dẫn đúng số trang (`page_no`) và tài liệu gốc (`source_doc`) (Mục tiêu: > 95%).
3. **Analytics Formula Integrity**: Xác thực chéo kết quả tính Z-Score, F-Score, DuPont giữa hàm Python và bảng tính Excel kế toán độc lập (Sai số chấp nhận: 0.0%).
4. **Latency Benchmark**:
   * Truy vấn Fact SQL: < 200ms
   * Chẩn đoán Sức khỏe (Z-Score/F-Score/DuPont): < 30ms
   * Dự báo Chuỗi thời gian: < 50ms
