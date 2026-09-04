# KIẾN TRÚC TẦNG SUY LUẬN (INFERENCE & SERVING ARCHITECTURE)

Tài liệu này đặc tả toàn bộ kiến trúc tầng suy luận (Inference Layer) của Hệ thống Trợ lý Tài chính: từ khâu tiếp nhận câu hỏi tự nhiên tiếng Việt, điều hướng ý định (Intent Routing), sinh câu lệnh Text-to-SQL, thực thi kiểm soát an toàn (Execution Sandbox), đến định dạng câu trả lời kèm trích dẫn nguồn minh bạch (Grounding & Provenance).

---

## 1. TỔNG QUAN LUỒNG SUY LUẬN (END-TO-END FLOW)

Hệ thống hoạt động theo triết lý **Structured-First Hybrid RAG**: ưu tiên tối đa tính toán số học trên Cơ sở dữ liệu quan hệ (đảm bảo độ chính xác 100% và Zero Hallucination), kết hợp cùng bộ đọc văn bản bán cấu trúc (FTS5 Document Reader) cho các câu hỏi định tính.

```
                          [Câu hỏi của người dùng]
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Intent & Entity Router (Bộ phân tích ngữ nghĩa & thực thể)              │
│    • Nhận diện thực thể tài chính: Ticker, Năm, Báo cáo mẹ/Hợp nhất, Kỳ     │
│    • Phân luồng ý định: Định lượng (Số học) vs Định tính (Chính sách/Ý kiến)│
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼ (90% Câu hỏi Số liệu)                 ▼ (10% Câu hỏi Thuyết minh/Văn bản)
┌────────────────────────────────────────┐ ┌────────────────────────────────────────┐
│ 2A. Text-to-SQL Engine                 │ │ 2B. Qualitative Document Reader        │
│ • Model: Qwen3.5-4B-SQL Local          │ │ • Truy vấn SQLite Native FTS5          │
│ • Kỹ thuật: Native Chain-of-Thought    │ │ • Bốc tách các đoạn thuyết minh/văn bản│
│   (<think>...</think>)                 │ │   theo số trang (page_no)              │
│ • Sinh câu lệnh SQL ANSI chuẩn xác     │ │ • Trích xuất nguyên văn ngữ cảnh       │
└──────────────────┬─────────────────────┘ └───────────────────┬────────────────────┘
                   │                                           │
                   ▼                                           ▼
┌────────────────────────────────────────┐                     │
│ 3. SQL Safety Guard & Execution Engine │                     │
│ • Kiểm tra an toàn: Read-Only (SELECT) │                     │
│ • Thực thi trên SQLite B-Tree Index    │                     │
│ • Cơ chế tự sửa lỗi (Self-Correction)  │                     │
└──────────────────┬─────────────────────┘                     │
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. Grounding & Response Generation Agent                                    │
│    • Định dạng số học tài chính chuẩn (Quy đổi Tỷ VNĐ / Triệu VNĐ)          │
│    • Đóng gói trích dẫn nguồn minh bạch: Tên tài liệu, Bảng, Số trang       │
│    • Trả lời người dùng với độ tin cậy tuyệt đối                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. CÁC THÀNH PHẦN CỐT LÕI (CORE COMPONENTS)

### 2.1. Intent & Entity Router (Bộ điều phối ý định & thực thể)
Nhiệm vụ: Phân tích cú pháp câu hỏi người dùng trước khi gọi đến mô hình ngôn ngữ lớn:
* **Bóc tách thực thể cứng (Deterministic Entity Extraction)**:
  * **Mã chứng khoán (Ticker)**: Ánh xạ từ danh mục 100 mã cổ phiếu (`code_stock.csv`) hoặc tên đầy đủ của doanh nghiệp (ví dụ: *"Vietjet"* $\rightarrow$ `VJC`, *"Hòa Phát"* $\rightarrow$ `HPG`).
  * **Năm tài chính (Fiscal Year)**: Regex nhận diện các năm từ `2015` đến `2025`.
  * **Phạm vi báo cáo (Report Scope)**:
    * Chứa từ khóa *"công ty mẹ"*, *"riêng lẻ"* $\rightarrow$ `report_type = 'separate'`
    * Mặc định hoặc chứa *"hợp nhất"* $\rightarrow$ `report_type = 'consolidated'`
* **Phân loại nhánh xử lý (Branch Routing)**:
  * **Nhánh Số liệu (Quantitative)**: Câu hỏi chứa các chỉ tiêu số học, hỏi *"bao nhiêu"*, *"so sánh"*, *"tăng trưởng"*, *"tổng"* $\rightarrow$ Chuyển sang **Text-to-SQL Engine**.
  * **Nhánh Định tính (Qualitative)**: Câu hỏi về *"ý kiến kiểm toán"*, *"nguyên tắc kế toán"*, *"chính sách cổ tức"*, *"rủi ro"* $\rightarrow$ Chuyển sang **Document Reader**.

---

### 2.2. Text-to-SQL Generation Engine (Động cơ sinh SQL)

Mô hình cốt lõi: **`Qwen3.5-4B-SQL`** (đã fine-tune trên 17.000 mẫu truy vấn tài chính nâng cao, hỗ trợ suy luận nội tại `<think>`).

#### Chiến lược Schema Prompting tối giản (Minimal Context Injection)
Thay vì nhồi nhét toàn bộ schema phức tạp vào prompt, hệ thống chỉ cung cấp đúng lược đồ bảng đích:
```text
Table: financial_facts
Columns:
- ticker: Mã cổ phiếu (VD: 'VJC', 'ACB')
- year: Năm báo cáo (VD: 2018, 2022)
- report_type: 'consolidated' (hợp nhất) hoặc 'separate' (công ty mẹ)
- statement: 'balance_sheet', 'income_statement', 'cash_flow', 'notes'
- item_name: Tên chỉ tiêu có dấu
- item_name_ascii: Tên chỉ tiêu không dấu viết thường (VD: 'lai tien gui')
- period_label: Nhãn kỳ (VD: 'Số cuối năm', '31/12/2022')
- value_vnd: Giá trị số thực quy về VNĐ
- raw_value: Giá trị chuỗi gốc
- page_no: Số trang tham chiếu
- source_doc: Tên tài liệu BCTC
```

#### Quy tắc sinh SQL bắt buộc (Model Constraints):
1. **Luôn sử dụng toán tử tìm kiếm mờ không dấu**: Dùng `item_name_ascii LIKE '%...%'` thay vì so khớp chính xác có dấu để chống sai lệch chính tả và từ đồng nghĩa.
2. **Quy đổi đơn vị trực tiếp trong SQL**:
   * Hỏi theo *"triệu đồng"*: `value_vnd / 1000000.0`
   * Hỏi theo *"tỷ đồng"*: `value_vnd / 1000000000.0`
3. **Luôn chọn kèm siêu dữ liệu dẫn nguồn**: Mọi câu `SELECT` đều phải lấy kèm `raw_value`, `page_no`, `source_doc`, `section_title`.
4. **Giới hạn kết quả**: Mặc định đặt `LIMIT 5` hoặc `LIMIT 1` để chống tràn bộ nhớ.

---

### 2.3. SQL Safety Guard & Execution Sandbox (Tầng kiểm soát an toàn CSDL)

Để đảm bảo tính toàn vẹn và an toàn tuyệt đối cho CSDL doanh nghiệp:
* **Quyền hạn chỉ đọc (Read-Only Enforcement)**:
  * CSDL SQLite được mở ở chế độ `immutable=1` hoặc sử dụng connection có cờ `read-only`.
  * Bộ lọc tĩnh (Static AST Validator) chặn đứng mọi câu lệnh chứa: `DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`, `PRAGMA`, `ATTACH`.
* **Giới hạn thời gian (Execution Timeout)**: Mọi truy vấn SQL bị ngắt cưỡng chế nếu thực thi quá `500ms`.
* **Cơ chế tự sửa lỗi & Tìm kiếm mờ dự phòng (Self-Correction & FTS5 Fallback)**:
  * Nếu câu lệnh SQL thực thi trả về 0 dòng kết quả (Empty Result Set), hệ thống tự động kích hoạt bảng ảo `facts_fts` (SQLite Full-Text Search) để tìm kiếm các chỉ tiêu kế toán có phát âm hoặc từ khóa tương tự, sau đó thử lại truy vấn.

---

### 2.4. Qualitative Document Reader (Nhánh đọc tài liệu định tính)

Dành cho 10% các câu hỏi không thể trả lời bằng con số đơn thuần:
* Sử dụng bảng chỉ mục toàn văn `facts_fts` kết hợp với văn bản thô theo trang:
  * Truy xuất trực tiếp các trang BCTC liên quan (ví dụ: Trang 3-6 chứa Báo cáo kiểm toán độc lập, Trang 10-15 chứa Chính sách kế toán).
  * Trích xuất các đoạn văn bản tương ứng với số trang và đưa vào ngữ cảnh để LLM tóm tắt câu trả lời cho người dùng.

---

### 2.5. Grounding & Response Generation Agent (Tầng đóng gói câu trả lời)

Kết quả thô từ CSDL sẽ được chuyển thành câu trả lời tự nhiên, đáp ứng tiêu chuẩn kiểm toán:
* **Quy chuẩn hiển thị số liệu**:
  * Số tiền lớn được định dạng có dấu chấm phân cách hàng nghìn (ví dụ: `208.253,2 triệu đồng` hoặc `72.917,6 tỷ đồng`).
  * Tỷ lệ phần trăm làm tròn đến 2 chữ số thập phân (`14,25%`).
* **Đính kèm trích dẫn kiểm chứng bắt buộc (Mandatory Audit Provenance)**:
  Mỗi câu trả lời của trợ lý AI bắt buộc phải có phần dẫn nguồn bên dưới:
  ```markdown
  📌 Nguồn dữ liệu kiểm chứng:
  • Doanh nghiệp: CTCP Hàng không Vietjet (VJC)
  • Tài liệu: Báo cáo tài chính kiểm toán công ty mẹ năm 2018 (VJC_financial_statements_2018_separate)
  • Vị trí: Trang 10 (Báo cáo lưu chuyển tiền tệ) và Trang 44 (Thuyết minh Doanh thu tài chính)
  • Số liệu gốc trong bảng: 208.253.201.298 VNĐ
  ```

---

## 3. MÔ HÌNH TRIỂN KHAI ĐA TẦNG (HYBRID DEPLOYMENT ARCHITECTURE)

Hệ thống được thiết kế theo kiến trúc **Tiered Serving** linh hoạt:

```
[Request từ Web / Chatbot / App]
                │
                ▼
┌────────────────────────────────────────────────────────┐
│ API Gateway / Serving Controller                       │
└───────────────────────┬────────────────────────────────┘
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│ Tier 1: Local SLM Engine │  │ Tier 2: Cloud Fallback   │
│ (Qwen3.5-4B-SQL GGUF)    │  │ (DeepSeek / GLM / Gemini)│
│ • Chạy qua Ollama/vLLM   │  │ • Dành cho phân tích vĩ  │
│ • Độ trễ: < 200ms        │  │   mô, bài luận chuyên sâu│
│ • Chi phí: 0 VNĐ         │  │ • Kích hoạt khi cần thiết│
│ • 100% Offline, Bảo mật  │  │                          │
└──────────────────────────┘  └──────────────────────────┘
```

1. **Tier 1 (Mặc định - Local On-Premise)**:
   * Phục vụ 85% - 90% nhu cầu tra cứu số liệu hàng ngày.
   * Model `Qwen3.5-4B-SQL` chạy cục bộ thông qua Ollama hoặc llama.cpp: không tốn chi phí token, không lo rò rỉ dữ liệu tài chính nội bộ.
2. **Tier 2 (Mở rộng - Cloud Reasoning API)**:
   * Khi người dùng yêu cầu viết báo cáo phân tích chiến lược, so sánh ngành hoặc phân tích vĩ mô tổng hợp: Hệ thống lấy dữ liệu số đã xác thực từ SQLite rồi gửi sang các model Cloud lớn (DeepSeek-V3, GLM-4) để tổng hợp thành văn bản phân tích chuyên sâu.

---

## 4. MA TRẬN ĐÁNH GIÁ CHẤT LƯỢNG (EVALUATION MATRIX)

Hệ thống đo lường hiệu năng dựa trên 4 chỉ số cốt lõi:
1. **Execution Accuracy (EX-Acc)**: Tỷ lệ câu SQL sinh ra thực thi thành công và cho ra kết quả số học trùng khớp 100% với báo cáo kiểm toán.
2. **Provenance Recall**: 100% câu trả lời đều phải truy xuất đúng số trang (`page_no`) và tên tài liệu gốc.
3. **Query Latency**: Thời gian xử lý từ lúc nhận câu hỏi đến khi có kết quả phải đạt dưới **1.0 giây** trên phần cứng tiêu chuẩn.
4. **Hallucination Rate**: Giữ ở mức **0.0%** đối với mọi con số tài chính nhờ tính toán trực tiếp từ cơ sở dữ liệu.
