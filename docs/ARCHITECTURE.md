# KIẾN TRÚC HỆ THỐNG TRỢ LÝ TÀI CHÍNH TEXT-TO-SQL GROUNDED RAG

> **Trạng thái triển khai**:
> * **Tầng Dữ liệu & ETL Pipeline**: `[ĐÃ TRIỂN KHAI / IMPLEMENTED]` (Mã nguồn: `scripts/build_db.py`, CSDL: `data/financial.db` gồm 2.116.243 bản ghi từ 1.973 báo cáo tài chính của 100 doanh nghiệp niêm yết).
> * **Tầng Suy luận & Phục vụ (Inference & Serving)**: `[THIẾT KẾ MỤC TIÊU / TARGET SPEC]` (Đặc tả chi tiết tại `docs/INFERENCE_ARCHITECTURE.md`).

---

## 1. TỔNG QUAN HỆ THỐNG (SYSTEM OVERVIEW)

Hệ thống được thiết kế theo mô hình **Structured-First Hybrid Financial Assistant**: kết hợp giữa tính toán tất định (Deterministic Execution) trên Cơ sở dữ liệu quan hệ và năng lực dịch ngôn ngữ tự nhiên của Mô hình ngôn ngữ lớn chuyên biệt (Text-to-SQL SLM).

```
[Người dùng đặt câu hỏi tự nhiên tiếng Việt]
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│ 1. Natural Language Interface                          │
│    - Tiếp nhận câu hỏi tài chính bằng tiếng Việt       │
│    - Chuẩn hóa thực thể (Mã CK, Năm, Loại BCTC, Kỳ)    │
└────────────────────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│ 2. Text-to-SQL Engine                                  │
│    - Base Model: Qwen/Qwen3.5-4B                      │
│    - Fine-tuned Adapter: giangkh19/qwen3.5-4b-sql      │
│    - Sinh chuỗi suy luận (<think> ... </think>)        │
│    - Dịch câu hỏi thành câu lệnh SQL ANSI chuẩn xác    │
└────────────────────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│ 3. Database Engine (SQLite Native + FTS5)              │
│    - Thực thi truy vấn B-Tree Index (< 2ms)            │
│    - Tìm kiếm toàn văn chỉ tiêu bằng SQLite FTS5       │
│    - Trả về số thực chuẩn hóa (VNĐ) kèm trích dẫn gốc  │
└────────────────────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│ 4. Grounded Response & Provenance                      │
│    - Định dạng số chuẩn (Tỷ / Triệu VNĐ)               │
│    - Đính kèm minh chứng kiểm toán: Tài liệu, Trang BCTC│
└────────────────────────────────────────────────────────┘
```

---

## 2. KIẾN TRÚC TẦNG DỮ LIỆU & ETL PIPELINE

### 2.1. Nguyên tắc thiết kế cốt lõi (Core Design Principles)
1. **Deterministic over Probabilistic**: Tuyệt đối không dùng LLM để bóc tách số liệu hay parse bảng. Toàn bộ khâu ETL sử dụng logic tất định (Regex, Rule-based Heuristics trong `scripts/build_db.py`) nhằm đảm bảo chi phí 0đ, tốc độ tối đa (259 giây cho ~2.000 file) và độ chính xác số học 100%.
2. **Provenance & Traceability**: Mọi con số trong hệ thống đều phải lưu vết ngược về nguồn gốc: Số trang (`page_no`), Tên tài liệu (`source_doc`), Phân mục (`section_title`), và Chuỗi gốc trong bảng (`raw_value`).
3. **Hỗ trợ 3 Chuẩn mực Kế toán Việt Nam**:
   * **Doanh nghiệp Sản xuất / Thương mại / Dịch vụ (76 mã)**: Thông tư 200/2014/TT-BTC (`[Chỉ tiêu, Mã số, Thuyết minh, Cuối năm, Đầu năm]`).
   * **Ngân hàng thương mại (21 mã)**: QĐ 479/2004/QĐ-NHNN & TT 49/2014/TT-NHNN (`Cho vay khách hàng`, `Tiền gửi KH`, `Thu nhập lãi thuần`, đơn vị gắn vào header cột).
   * **Công ty Chứng khoán (3 mã)**: Thông tư 210/2014/TT-BTC & TT 334/2016/TT-BTC (`[Mã số, CHỈ TIÊU, Thuyết minh, Cuối năm, Đầu năm]` - Mã số ở cột 0; chỉ tiêu `FVTPL`, `AFS`, `Doanh thu môi giới`).

---

### 2.2. Quy trình Trích xuất (ETL Ingestion Flow)

Quy trình triển khai thực tế trong `scripts/build_db.py`:

```
[Kho tài liệu BCTC OCR (1.973 files)] + [Danh mục mã chứng khoán (code_stock.csv)]
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ Bước 1: Metadata Resolution                                                   │
│ - Bóc tách Mã CK, Năm báo cáo, Loại BCTC từ đường dẫn.                        │
│ - Hỗ trợ đầy đủ 3 loại báo cáo (report_type):                                 │
│   • 'consolidated' (Báo cáo hợp nhất): 1.174.853 bản ghi                      │
│   • 'separate' (Báo cáo công ty mẹ / riêng lẻ): 933.557 bản ghi               │
│   • 'aggregated' (Báo cáo tổng hợp - ACV, DTK, SJG...): 7.833 bản ghi         │
│ - Ánh xạ tên đầy đủ doanh nghiệp từ code_stock.csv.                           │
└───────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ Bước 2: Page-Level Context & Unit Detection                                   │
│ - Tách văn bản theo các mốc trang (===== PAGE X =====).                       │
│ - Nhận diện đơn vị tiền tệ trang (VND, Triệu VND, Tỷ VND, Nghìn VND, USD).   │
│ - Phân loại Báo cáo (statement):                                              │
│   • 'balance_sheet' (Bảng cân đối kế toán): 215.510 bản ghi                   │
│   • 'income_statement' (Báo cáo kết quả HĐKD): 71.046 bản ghi                 │
│   • 'cash_flow' (Báo cáo lưu chuyển tiền tệ): 6.581 bản ghi                   │
│   • 'notes' (Thuyết minh BCTC): 1.823.106 bản ghi (86,1%)                     │
│   ⚠️ Lưu ý: Tỷ trọng 'cash_flow' độc lập thấp vì phần lớn chỉ tiêu lưu chuyển│
│      tiền tệ chi tiết nằm trong phần Thuyết minh ('notes').                   │
└───────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ Bước 3: Dynamic Table Parsing & Normalization                                 │
│ - Quét cấu trúc bảng HTML (<table>, <tr>, <td>).                              │
│ - Bộ nhận diện cột động (Dynamic Column Mapper):                              │
│   • Tự động phân định cột Tên chỉ tiêu, cột Mã số (item_code).                │
│   • Bóc tách các cột kỳ báo cáo (Số cuối năm, Số đầu năm, Năm nay, Năm trước).│
│   • Ghi đè đơn vị tính theo từng cột nếu có (ví dụ: '31.12.2022 Triệu VND').  │
│ - Chuẩn hóa số học kế toán:                                                   │
│   • Xử lý số âm trong ngoặc: (208.253.201.298) -> -208253201298.0.            │
│   • Dấu phân cách hàng nghìn / thập phân theo chuẩn kế toán Việt Nam.         │
│   • Quy đổi về VNĐ chuẩn đối với tiền đồng (value_vnd = raw_number * factor). │
│   • Giữ nguyên factor = 1.0 cho đơn vị USD (lưu kèm unit = 'USD').            │
└───────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ Bước 4: Batch Rebuild & Indexing                                              │
│ - Cơ chế hiện tại: Deterministic Batch Rebuild (DROP TABLE IF EXISTS và       │
│   nạp lô theo transaction 10.000 dòng để tối ưu tốc độ).                      │
│ - Tự động đồng bộ sang bảng ảo SQLite FTS5 qua Trigger AFTER INSERT.          │
│ - Đánh chỉ mục B-Tree đa cột cho truy vấn tức thì (< 2ms).                    │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. THIẾT KẾ CƠ SỞ DỮ LIỆU (DATABASE SCHEMA DESIGN)

CSDL lưu trữ tại [`data/financial.db`](file:///D:/GURU/data/financial.db) gồm **2 bảng duy nhất**:

### 3.1. Bảng Dữ liệu Tài chính (`financial_facts` - Đầy đủ 15 cột thực tế)

```sql
CREATE TABLE financial_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,              -- Mã CP: 'VJC', 'ACB', 'HPG', 'ACV'
    company_name TEXT,                 -- 'CTCP Hàng không Vietjet' (từ code_stock.csv)
    year INTEGER NOT NULL,             -- 2015 đến 2025
    report_type TEXT NOT NULL,         -- 'consolidated' | 'separate' | 'aggregated'
    statement TEXT NOT NULL,           -- 'balance_sheet' | 'income_statement' | 'cash_flow' | 'notes'
    section_title TEXT,                -- Tiêu đề phân mục / Thuyết minh BCTC kèm theo
    item_code TEXT,                    -- Mã số kế toán: '100', '110', '05', 'I', 'II'
    item_name TEXT NOT NULL,           -- Tên chỉ tiêu có dấu gốc ('Lãi tiền gửi và cho vay')
    item_name_ascii TEXT NOT NULL,     -- Tên không dấu ('lai tien gui va cho vay')
    period_label TEXT,                 -- 'Số cuối năm', '31/12/2022', '2018VND'
    value_vnd REAL,                    -- Giá trị số thực quy chuẩn về VNĐ
    raw_value TEXT,                    -- Chuỗi số gốc trong bảng OCR: '(208.253.201.298)'
    unit TEXT,                         -- 'VND', 'Triệu VND', 'USD'
    page_no INTEGER,                   -- Số trang chính xác trong tài liệu BCTC
    source_doc TEXT NOT NULL           -- Tên file: 'VJC_financial_statements_2018_separate'
);

-- Chỉ mục B-Tree tối ưu truy vấn Text-to-SQL
CREATE INDEX idx_facts_core ON financial_facts (ticker, year, report_type);
CREATE INDEX idx_facts_ascii ON financial_facts (item_name_ascii);
CREATE INDEX idx_facts_stmt ON financial_facts (statement);
```

### 3.2. Bảng Tìm kiếm Toàn văn (`facts_fts` - SQLite FTS5)

```sql
CREATE VIRTUAL TABLE facts_fts USING fts5(
    item_name,
    item_name_ascii,
    section_title,
    content='financial_facts',
    content_rowid='id'
);
```

---

## 4. KIỂM THỬ XÁC MINH HỆ THỐNG (VERIFICATION SUITE)

Hệ thống tích hợp hàm kiểm thử tự động `verify_database()` đối chiếu trực tiếp kết quả bóc tách với các câu hỏi thực tế trong `questions.jsonl`:

1. **Kiểm tra quy mô CSDL**: Tổng số bản ghi đạt **2.116.243 facts** trên 100/100 mã cổ phiếu.
2. **Kiểm chứng Câu 1 (Vietjet - VJC 2018 công ty mẹ)**:
   * Câu hỏi: *"Lãi tiền gửi năm 2018 của công ty mẹ Vietjet (VJC) là bao nhiêu triệu đồng?"*
   * Kết quả: Trích xuất chính xác **`208.253,2 triệu đồng`** (`208.253.201.298` VNĐ) tại Trang 10 (Lưu chuyển tiền tệ) và Trang 44 (Thuyết minh DT tài chính) của file `VJC_financial_statements_2018_separate`.
3. **Kiểm chứng Câu 2 (Ngân hàng ACB 2022 công ty mẹ)**:
   * Câu hỏi: *"Số dư cho vay khách hàng ngành Thương mại của công ty mẹ ACB cuối năm 2022 là bao nhiêu triệu đồng?"*
   * Kết quả: Trích xuất chính xác **`72.917.566 triệu đồng`** tại Trang 42 (Thuyết minh 9: Phân tích cho vay theo ngành) của file `ACB_financial_statements_2022_separate`.

---

## 5. TÍNH TƯƠNG THÍCH VÀ CƠ CHẾ NẠP DỮ LIỆU TƯƠNG LAI

1. **Cơ chế nạp hiện tại vs Lộ trình mở rộng**:
   * **Hiện tại (Batch Rebuild)**: Pipeline chạy toàn bộ 1.973 file theo lô trong 4.3 phút. Phù hợp cho việc chuẩn hóa định kỳ dữ liệu BCTC cả thị trường.
   * **Lộ trình (Incremental Ingestion)**: Khi người dùng tải lên tài liệu mới qua Chatbot, hệ thống sẽ chạy parser trên file đơn lẻ và thực hiện `INSERT` bổ sung vào bảng `financial_facts` mà không cần rebuild toàn bộ CSDL.
2. **Phạm vi nền tảng CSDL (Database Compatibility)**:
   * CSDL hiện tại được tối ưu hóa 100% cho **SQLite Native** (nhúng cục bộ, FTS5 virtual table, Triggers, WAL mode, Zero-ops).
   * **Chiến lược di chuyển (Migration Strategy)**: Toàn bộ bảng dữ liệu `financial_facts` tuân thủ chuẩn ANSI SQL. Khi chuyển đổi sang PostgreSQL (cho môi trường SaaS đa người dùng), chỉ cần thay thế bảng ảo FTS5 bằng extension `pg_trgm` / `tsvector` và viết lại cú pháp trigger.
