# KIẾN TRÚC HỆ THỐNG TRỢ LÝ TÀI CHÍNH TEXT-TO-SQL GROUNDED RAG

Tài liệu này mô tả toàn bộ kiến trúc tổng thể, thiết kế tầng dữ liệu (Data & ETL Pipeline) và cơ chế suy luận phục vụ truy vấn tài chính chính xác tuyệt đối (Zero Hallucination).

---

## 1. TỔNG QUAN HỆ THỐNG (SYSTEM OVERVIEW)

Hệ thống được thiết kế theo mô hình **Hybrid Enterprise Financial Assistant**: kết hợp giữa tính toán tất định (Deterministic Execution) của Cơ sở dữ liệu quan hệ và năng lực hiểu ngôn ngữ tự nhiên của Mô hình ngôn ngữ lớn (Text-to-SQL LLM).

```
[Người dùng đặt câu hỏi]
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 1. Natural Language Interface                          │
│    - Tiếp nhận câu hỏi tài chính bằng tiếng Việt       │
│    - Chuẩn hóa thực thể (Mã CK, Năm, Loại BCTC)        │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 2. Text-to-SQL Model (Qwen3.5-4B-SQL Fine-tuned)       │
│    - Sinh chuỗi suy luận nội tại (<think> ... </think>)│
│    - Dịch câu hỏi thành câu lệnh SQL ANSI chuẩn xác    │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 3. Database Engine (SQLite / PostgreSQL)               │
│    - Thực thi truy vấn B-Tree Index (< 2ms)            │
│    - Tìm kiếm mờ chỉ tiêu bằng SQLite Native FTS5      │
│    - Trả về số thực chuẩn hóa (VND) kèm trích dẫn gốc  │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 4. Grounded Response & Provenance                      │
│    - Định dạng số (Tỷ / Triệu đồng)                   │
│    - Đính kèm minh chứng: Tên tài liệu, Bảng, Số trang │
└────────────────────────────────────────────────────────┘
```

---

## 2. KIẾN TRÚC TẦNG DỮ LIỆU & ETL PIPELINE

### 2.1. Nguyên tắc thiết kế (Core Design Principles)
1. **Deterministic over Probabilistic**: Tuyệt đối không dùng LLM để bóc tách số liệu hay parse bảng. Toàn bộ khâu ETL sử dụng logic tất định (Regex, Rule-based Heuristics) nhằm đảm bảo chi phí 0đ, tốc độ tối đa và độ chính xác số học 100%.
2. **Provenance & Traceability**: Mọi con số trong hệ thống đều phải lưu vết ngược về nguồn gốc gốc (Số trang `page_no`, Tên tài liệu `source_doc`, Chuỗi gốc trong bảng `raw_value`).
3. **Adaptive Schema Support**: Hỗ trợ 3 hệ thống chuẩn mực kế toán Việt Nam khác biệt về cấu trúc cột:
   * **Doanh nghiệp Sản xuất / Dịch vụ**: Cấu trúc Thông tư 200/2014/TT-BTC.
   * **Ngân hàng thương mại**: Cấu trúc QĐ 479/2004/QĐ-NHNN & TT 49/2014/TT-NHNN.
   * **Công ty Chứng khoán**: Cấu trúc Thông tư 210/2014/TT-BTC & TT 334/2016/TT-BTC.

---

### 2.2. Quy trình trích xuất thích ứng (Adaptive Ingestion Flow)

```
[Kho tài liệu BCTC OCR (1.973 files)] + [Danh mục mã chứng khoán (code_stock.csv)]
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ Bước 1: Metadata Resolution                                                   │
│ - Trích xuất Mã CK, Năm báo cáo, Loại BCTC (Hợp nhất / Riêng) từ đường dẫn.  │
│ - Ánh xạ tên đầy đủ doanh nghiệp từ code_stock.csv.                           │
└───────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ Bước 2: Page-Level Context & Unit Detection                                   │
│ - Tách văn bản theo các mốc trang (===== PAGE X =====).                       │
│ - Nhận diện đơn vị tiền tệ trang (VND, Triệu VND, Tỷ VND, Nghìn VND, USD).   │
│ - Phân loại Báo cáo: Bảng cân đối kế toán, Kết quả HĐKD, Lưu chuyển tiền tệ,  │
│   hoặc Thuyết minh BCTC dựa trên tiêu đề ngữ cảnh.                            │
└───────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ Bước 3: Dynamic Table Parsing                                                 │
│ - Quét cấu trúc bảng HTML (<table>, <tr>, <td>).                              │
│ - Bộ nhận diện cột động (Dynamic Column Mapper):                              │
│   • Phân định cột Tên chỉ tiêu, cột Mã số (item_code).                        │
│   • Bóc tách các cột kỳ báo cáo (Số cuối năm, Số đầu năm, Năm nay, Năm trước).│
│   • Tự động kế thừa hoặc ghi đè đơn vị tiền tệ từ tiêu đề cột.                │
│ - Chuẩn hóa số học kế toán:                                                   │
│   • Xử lý số âm trong ngoặc: (X) -> -X.                                       │
│   • Chuẩn hóa dấu chấm/phẩy theo quy chuẩn kế toán Việt Nam.                  │
│   • Quy đổi toàn bộ giá trị về VNĐ chuẩn (value_vnd = raw_number * factor).   │
└───────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ Bước 4: Batch Persistence & Indexing                                          │
│ - Nạp theo lô (Batch Insert) vào bảng financial_facts trong SQLite.           │
│ - Đồng bộ tự động sang bảng ảo SQLite FTS5 (Full-Text Search).                │
│ - Tạo chỉ mục B-Tree (Composite Index) phục vụ truy vấn tức thì.              │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. THIẾT KẾ CƠ SỞ DỮ LIỆU (DATABASE SCHEMA DESIGN)

### 3.1. Bảng Dữ liệu Tài chính (`financial_facts`)
Lưu trữ toàn bộ các chỉ tiêu số học đã chuẩn hóa cùng siêu dữ liệu dẫn nguồn:

* **Định danh & Thực thể**:
  * `id`: Khóa chính tự tăng.
  * `ticker`: Mã cổ phiếu (ví dụ: `VJC`, `ACB`, `HPG`).
  * `company_name`: Tên công ty đầy đủ.
  * `year`: Năm tài chính.
  * `report_type`: Loại báo cáo (`consolidated` - Hợp nhất, `separate` - Công ty mẹ).
  * `statement`: Phân loại báo cáo (`balance_sheet`, `income_statement`, `cash_flow`, `notes`).
  * `section_title`: Tiêu đề phân mục hoặc tên mục thuyết minh kèm theo.

* **Chỉ tiêu kế toán & Giá trị**:
  * `item_code`: Mã số kế toán (ví dụ: `100`, `110`, `05`).
  * `item_name`: Tên chỉ tiêu có dấu gốc (ví dụ: `Lãi tiền gửi và cho vay`).
  * `item_name_ascii`: Tên chỉ tiêu chuẩn hóa không dấu (ví dụ: `lai tien gui va cho vay`).
  * `period_label`: Nhãn kỳ kế toán (ví dụ: `Số cuối năm`, `31/12/2022`).
  * `value_vnd`: Giá trị số thực quy chuẩn về VNĐ (kiểu `REAL`).
  * `raw_value`: Chuỗi số gốc trong bảng (ví dụ: `(208.253.201.298)`).
  * `unit`: Đơn vị tiền tệ gốc của bảng.

* **Trích dẫn minh bạch (Grounding / Audit Trail)**:
  * `page_no`: Số trang chính xác trong tài liệu BCTC PDF/OCR.
  * `source_doc`: Tên định danh của tài liệu gốc.

### 3.2. Bảng tìm kiếm toàn văn (`facts_fts`)
* Sử dụng module `FTS5` tích hợp sẵn trong SQLite.
* Hỗ trợ tìm kiếm từ khóa chỉ tiêu kế toán nhanh chóng, xử lý tốt sự sai lệch câu chữ giữa câu hỏi người dùng và tên gọi kế toán chuẩn mực.

---

## 4. CƠ CHẾ MỞ RỘNG (EXTENSIBILITY FOR NEW IMPORTS)

Hệ thống được thiết kế để mở rộng tiếp nhận tài liệu mới từ người dùng qua các giai đoạn:
1. **File Drop / Upload**: Tiếp nhận tài liệu dạng Text OCR hoặc PDF.
2. **Signature Classifier**: Nhận diện loại mẫu biểu dựa trên từ khóa đặc trưng (Ngân hàng / Chứng khoán / Doanh nghiệp chung).
3. **Append-only Ingestion**: Chạy parser và nạp thẳng các bản ghi mới vào CSDL hiện có mà không làm gián đoạn hệ thống.
4. **Database-Agnostic Support**: Toàn bộ lược đồ và câu lệnh SQL tuân thủ chuẩn ANSI SQL, cho phép chuyển đổi sang PostgreSQL hoặc DuckDB chỉ bằng cách thay đổi cấu hình kết nối.
