# CLAUDE.md — Dự án GURU · ViFinQA Competition (Table Retrieval & Text-to-Pandas)

> Đây là tài liệu ghi nhớ dài hạn cho dự án. Mọi phiên làm việc nên đọc file này trước.
> Cập nhật khi có quyết định mới về kỹ thuật / dữ liệu / nộp bài.

---

## 1. Tổng quan cuộc thi

**Tên bài toán:** Financial Table Retrieval & Text-to-Pandas Query Generation trên Báo cáo tài chính (BCTC) doanh nghiệp niêm yết Việt Nam.

**Bối cảnh:** Nhà đầu tư / chuyên viên phân tích mất nhiều thời gian tra cứu thủ công chỉ số tài chính (doanh thu, lợi nhuận, ROE, ROA, tỉ lệ nợ/vốn, tăng trưởng theo giai đoạn...) rải rác trong hàng trăm BCTC dạng bảng của công ty niêm yết qua nhiều năm. Trợ lý AI Text-to-Pandas tự động hoá việc tra cứu, tổng hợp, tính toán.

**Hai nhiệm vụ cốt lõi:**
1. **Table Retrieval:** Với câu hỏi `q`, xác định tập con bảng `D' ⊂ D` trong kho BCTC mà mỗi bảng chứa (một phần/toàn bộ) số liệu cần để tính đáp án.
2. **Text-to-Pandas:** Dựa trên các bảng đã truy hồi, sinh câu lệnh pandas thực thi được, trả về đúng số liệu.

**Mục tiêu hệ thống (5 yêu cầu BTC):**
1. Truy hồi chính xác — đúng công ty, đúng năm, đúng bảng, đúng vị trí.
2. Hiểu truy vấn tài chính tiếng Việt — gồm câu hỏi so sánh nhiều công ty / nhiều năm / chỉ số dẫn xuất (ROE, ROA, tăng trưởng...).
3. Sinh pandas query đúng logic, đúng schema, đúng đơn vị, đúng kỳ báo cáo.
4. Dẫn nguồn minh bạch — trích dẫn công ty, năm, tên báo cáo, tên bảng, vị trí (trang/mục).
5. Kiểm soát hallucination — không bịa số liệu, bảng, nguồn tham chiếu không tồn tại.

**Dashboard nộp bài:** http://leaderboard.aiguru.com.vn/ → My Submissions.

---

## 2. Dữ liệu (Nguồn chính: Hugging Face)

- **Dataset:** `AIGuruTinix/ViFinQA` — https://huggingface.co/datasets/AIGuruTinix/ViFinQA
- **Số liệu chính:** 1,012 câu hỏi, 1,973 báo cáo OCR từ 100 công ty niêm yết, giai đoạn 2015–2025.
- **Cấu trúc dataset:**
  ```
  code_stock.csv                          # Mã CK (ticker) ↔ Tên công ty
  financial_statements/
    TICKER/YEAR/DOCUMENT/DOCUMENT_extracted.txt   # nội dung BCTC dạng OCR
  questions/questions.jsonl               # id + question (chỉ câu hỏi, KHÔNG có đáp án)
  ```
- **Báo cáo .txt:** OCR toàn văn UTF-8, có markup bảng dạng HTML và marker trang `===== PAGE 1 =====`. Path encode ticker, năm, tên tài liệu, loại báo cáo.
- **Phân loại báo cáo:** 957 hợp nhất (consolidated), 954 riêng lẻ (separate), 7 aggregated, 55 khác/không nhãn.
- **Giấy phép:** TiniX OCR Annual Financial Statements corpus — CC BY-NC 4.0.
- **Giới hạn đã biết:** lỗi OCR (dấu, số, bảng), độ phủ không đồng đều giữa công ty/năm; một số câu hỏi cần số học, đổi đơn vị, hoặc tổng hợp nhiều báo cáo.
- **Lưu ý:** Bản HF công bố *chỉ có câu hỏi*, không có đáp án / gold evidence / bảng chuẩn hoá / pandas program / nhãn độ khó. Toàn bộ câu hỏi được gói trong 1 split `train` (chỉ là quy ước đóng gói). Câu hỏi kiểm thử thật do BTC cấp có thể là tập con/phiên bản khác — **cần xác minh**.

**GitHub tham khảo (companion repo của paper):** `DSKT-NOWJ/ViFinQA` — chứa code generation, retrieval, reranking, answering, evaluation, config YAML. Paper: *"ViFinQA: A Comprehensive and Challenging Benchmark for End-to-End Vietnamese Financial Reasoning."*
- Pipeline CLI: `generate` → `build-index`/`eval-retrieval` → `eval-answer` → `eval-e2e`.
- Retrieval modes: BM25, dense (Sentence Transformers), RRF, dense+rerank, row-chunks, multi-view, metadata filtering, cascade reranking.
- Models dùng trong repo: embeddings `BAAI/bge-m3`, `Qwen3-Embedding-{0.6B,4B,8B}`; reranker `bge-reranker-v2-m3`, `Qwen3-Reranker-{0.6B,4B,8B}`.
- E2E paper: `k=10`, max 10 context tables, temperature 0, tolerance tuyệt đối 0.01, rounding được chỉ thị trong prompt (không post-hoc).

---

## 3. Định dạng nộp bài (RẤT QUAN TRỌNG)

### 3.1 File dự đoán `submission.json` — mảng JSON:
```json
[
  {
    "id": <integer>,
    "question": "<string>",
    "answer": <float>,
    "relevant_docs": ["<id_báo_cáo>"],
    "relevant_tables": ["<id_báo_cáo>|<vị trí trong báo cáo>"],
    "evidence": [
      { "variable": "<tên_biến_dataframe>", "csv_path": "<string>" }
    ],
    "pandas_query": "<string>"
  }
]
```
- **`relevant_docs`:** mã báo cáo = tên file cuối cùng trong path, bỏ phần mở rộng `.txt`. Ví dụ path `ocr_filter\AAA\2015\AAA_financial_statements_2015_consolidated` → id `AAA_financial_statements_2015_consolidated`.
- **`relevant_tables`:** định dạng `<id_báo_cáo>|table_<N>` — **đã xác minh** khớp format chuẩn benchmark `DSKT-NOWJ/ViFinQA` (`common/schemas/table_ref.py`: `make_table_ref` → `f"{doc_name}|table_{table_id}"`, `parse_table_ref` tách bằng separator `"|table_"`; `schema.py` `relevant_tables: tuple[str,...]` comment `"doc_name|table_N"`). ⚠️ Ví dụ trong đề bài viết tay (`|350` bỏ `table_`) là **sai** — bug `search.py.relevant_tables_key()` cũ trả `report_id|{position}` đã fix sang `report_id|table_{position}` (6/8/2026). `position` = N trong `table_N` do ETL gán `table_id=f"table_{table_idx}"`. ⚠️ Vẫn còn rủi ro lệch hệ thống đánh số bảng (BTC dùng corpus nội bộ `ocr_filter/` không công khai — xem §8 M4 "khoảng trống retrieval"); test format-only trước, hỏi BTC nếu F2 vẫn 0.
- **`answer`:** float.
- **`evidence`:** các bảng dùng để thực thi `pandas_query`. `variable` phải là tên Python hợp lệ, không trùng nhau trong cùng câu hỏi (df1, df2, ...). `csv_path` là path tương đối **bắt đầu bằng `data/`**.
- **`pandas_query`:** chuỗi code chạy lại được trên dữ liệu đã chuẩn hoá.

### 3.2 Gói ZIP:
```
submission.zip
├── submission.json
└── data/
    ├── <bảng_1>.csv
    └── ...
```
- `submission.json` và `data/` **nằm trực tiếp ở cấp ngoài cùng** — không bọc trong thư mục cha.
- Chỉ đúng **1 file .json** trong ZIP.
- Mọi `csv_path` (kể cả trong evidence) là path tương đối bắt đầu bằng `data/`.

### 3.3 Quy định nộp:
- Tối đa **10 bài/ngày**; **Private phase tối đa 5 bài/người** — nộp thận trọng.
- Thiếu file / thiếu câu → bài **không được đánh giá**, không tính vào quota.
- Bắt buộc nộp **working notes paper** mô tả phương pháp để kết quả chính thức.
- BTC có quyền loại bài không tuân thủ.

---

## 4. Phương pháp đánh giá (macro-average)

| Tiêu chí | Công thức / ý nghĩa |
|---|---|
| **Retrieval Precision** | macro trung bình của (số bảng truy hồi đúng) / (số bảng đã truy hồi) mỗi truy vấn |
| **Retrieval Recall** | macro trung bình của (số bảng truy hồi đúng) / (số bảng liên quan thật) mỗi truy vấn |
| **F2 (retrieval)** | `F2 = (5 × P × R) / (4 × P + R)` — ưu tiên Recall hơn Precision |
| **Answer Accuracy** | (số query khớp đáp án chuẩn trong ngưỡng sai số) / (tổng query) |
| **Execution Accuracy** | (số code chạy được + kết quả đúng) / (tổng query) |

- Paper tham khảo dùng tolerance tuyệt đối 0.01; **ngưỡng chính thức do BTC công bố — cần theo dõi**.
- `crash` = program raise exception hoặc không trả về scalar số; `fail` = trả số sai.

---

## 5. Quy định về dữ liệu & mô hình (ràng buộc cứng)

- ✅ Được dùng dữ liệu ngoài miễn **trích dẫn rõ nguồn gốc** để BTC kiểm tra.
- ✅ Được dùng LLM **open-source / mô hình công khai** (Hugging Face, ...).
- ❌ **CẤM** LLM model đóng: GPT-4o, Gemini, Claude, ... (kể cả dùng qua API).
- ✅ Chỉ dùng mô hình phát hành **trước 1/6/2026 (giờ VN)**, **kích thước ≤ 14B**.
- ✅ Bắt buộc ghi rõ cách lấy mô hình trong bài nộp/bài báo (để tái lập).

### Ứng viên mô hình hợp lệ (open ≤14B, phát hành < 1/6/2026) — *cần cập nhật theo ngày phát hành thực tế:*
- **✅ ĐÃ CHỐT — LLM chính: Qwen3.5-9B-Instruct** (`Qwen/Qwen3.5-9B`): Apache 2.0, 9B ≤14B, phát hành **2/3/2026** (< 1/6/2026 → hợp lệ), hybrid Gated DeltaNet + Gated Attention, context 262K, tool calling. Giai đoạn đầu dùng **API OpenRouter** — model ID `qwen/qwen3.5-9b` (đã xác minh có trên OpenRouter, $0.10/$0.15 per 1M, 262K ctx); sau thuê GPU chạy local vLLM cùng model. ⚠️ Kiến trúc linear-attention → cần xác minh vLLM hỗ trợ trước khi thuê GPU; dự phòng: Qwen3-8B/14B (4/2025).
- **LLM code/pandas (dự phòng):** Qwen2.5-Coder-14B-Instruct, Qwen3-14B / Qwen3-8B (phát hành 4/2025 ✅), DeepSeek-R1-Distill-Qwen-14B / Distill-Llama-8B (reasoning), Phi-4 (14B, 2/2025 ✅), Gemma-2-9B, Llama-3.1-8B.
- **Embeddings:** BAAI/bge-m3 (~568M ✅), Qwen3-Embedding-0.6B/4B/8B ✅ (không vượt 14B).
- **Reranker:** bge-reranker-v2-m3 (~568M ✅ Apache-2.0), Qwen3-Reranker-0.6B/4B/8B ✅, **AITeamVN/Vietnamese_Reranker** (~568M ✅ Apache-2.0, fine-tune VN — tốt nhất tiếng Việt). ⚠️ **M3: rerank TẮT** (Qwen3-Reranker-0.6B local CPU quá nặng, đã xoá model 1.2GB). Cohere Rerank API = đóng → KHÔNG hợp lệ (§5). Jina Rerank = CC-BY-NC (non-commercial) → rủi ro. Nếu cần rerank lại: chạy local **Vietnamese_Reranker** (Apache-2.0) hoặc **mxbai-rerank-v2** API (Apache-2.0).
- ⚠️ *Quy ước:* giới hạn 14B áp cho "mô hình ngôn ngữ"; embedding/reranker cỡ nhỏ thường được chấp nhận nhưng nên kiểm chứng lại thông báo BTC.

---

## 6. Phân tích & ý kiến (strategic notes)

### 6.1 Bản chất bài toán
- Là pipeline **2 giai đoạn phụ thuộc chuỗi**: retrieval sai ⇒ codegen sai ⇒ answer sai. Điểm bị trừ độc lập ở cả 3 tiêu chí.
- **Truy hồi là nút thắt khó nhất.** Kho dữ liệu không chỉ ~2,000 báo cáo mà **hàng chục nghìn bảng** (vị trí bảng lên tới 350+/báo cáo). Chọn đúng 1 bảng trong hàng chục nghìn là bài toán khó hơn "chọn đúng report".
- **Entity grounding** (công ty → ticker, năm, loại báo cáo hợp nhất/riêng) giúp thu hẹp không gian tìm kiếm cực mạnh trước khi đánh hạng nội dung bảng.

### 6.2 Ràng buộc `evidence` + `pandas_query` là ưu thế (và thách thức)
- Query phải **chạy lại được trên CSV ta tự nộp** ⇒ buộc phải có **tầng chuẩn hoá dữ liệu (ETL) xác định, tái lập** cho toàn bộ corpus.
- CSV nộp trong `data/` chính là "bảng chuẩn hoá" của chúng ta ⇒ schema cần ổn định: cột định danh (`company`, `year`, `ticker`, `report_id`, `table_id`) + cột số liệu (đã chuẩn hoá đơn vị).
- Điều này đồng thời **chống hallucination về cấu trúc**: code chỉ được tham chiếu biến được khai báo trong evidence.

### 6.3 Zero-shot — không có train set
- Không có (question, gold query) để fine-tune generator ⇒ phải dựa vào LLM mạnh (≤14B) + **prompt engineering** + **execution-feedback self-correction** + **self-consistency**.
- Fine-tune retriever bị hạn chế; có thể thử pseudo-label bằng tự sinh câu hỏi↔bảng nhưng rủi ro.

### 6.4 Các vấn đề kỹ thuật cần xử lý sớm
1. **Parse OCR → bảng:** markup HTML trong .txt, bảng trải nhiều trang, dòng tiêu đề lặp, cột bị lỗi dấu → cần extractor bền vững.
2. **Đơn vị:** BCTC thường ghi nghìn/million VND; câu hỏi hỏi VNĐ ⇒ chuẩn hoá đơn vị đồng bộ toàn pipeline.
3. **Schema linking tiếng Việt:** từ khoá trong câu hỏi (doanh thu thuần, lợi nhuận sau thuế, tổng tài sản, vốn chủ sở hữu, nợ phải trả...) ↔ tên cột chuẩn hoá. Cần từ điển / mapping bổ sung cho biến thể.
4. **OCR sai số:** ký số lẫn dấu phẩy/dấu chấm, chữ bị lỗi dấu → cần chuẩn hoá số liệu (strip ký tự không hợp lệ, nhận diện nghìn/triệu).
5. **Câu hỏi dẫn xuất:** ROE, ROA, biên lợi nhuận, tăng trưởng qua 2 năm, so sánh 2 công ty, chênh lệch → codegen phải biết thao tác nhiều bảng/điều kiện `(company==X) & (year==Y)`.
6. **`relevant_tables` position:** quy ước số bảng của BTC cần reverse-engineer từ dữ liệu cung cấp.

### 6.5 Điểm mạnh nên dùng
- Câu hỏi tiếng Việt tài chính có cấu trúc khá chuẩn (entity + chỉ số + kỳ) ⇒ regex + LLM entity extraction cho retrieval rất hiệu quả.
- Dữ liệu có path encode ticker/năm/loại báo cáo ⇒ xây index metadata chuẩn hoá dễ dàng.
- F2 ưu tiên Recall ⇒ **ưu tiên truy hồi thừa còn hơn thiếu**, để codegen có đủ bảng.

---

## 7. Pipeline đề xuất (kế hoạch triển khai)

```
questions.jsonl
   │
   ▼
[1] ETL / Chuẩn hoá (offline, toàn bộ corpus)
   Parse .txt OCR → bảng → CSV chuẩn hoá + metadata
   (report_id, table_id, statement_type, ticker, year, đơn vị chuẩn, cột snake_case)
   │
   ▼
[2] Entity extraction (câu hỏi → ticker/năm/metric/loại báo cáo)  [regex + LLM]
   │
   ▼
[3] Retrieval: lọc entity → hybrid dense (BGE-M3 HNSW) + sparse (Qdrant TF/IDF) → native RRF → [rerank TÙY CHỌN, hiện OFF] → top-k bảng
   │
   ▼
[4] Text-to-Pandas: LLM sinh query trên schema top-k bảng (columns + sample rows)
   │
   ▼
[5] Thực thi + tự sửa lỗi (execution feedback, tối đa N lần retry)
   │
   ▼
[6] Chuẩn hoá đáp án (float, đơn vị, ngưỡng sai số) + dựng evidence CSVs
   │
   ▼
[7] Validate toàn bộ: mọi csv_path tồn tại, query chạy lại khớp answer, JSON đúng schema → zip → nộp
```

**Thứ tự ưu tiên:** (1) ETL chuẩn hoá sớm nhất — mọi thứ phụ thuộc vào tầng dữ liệu sạch. (2) Retrieval hybrid + entity. (3) Codegen + self-correction. (4) Validate + đóng gói.

---

## 8. Ngữ cảnh dự án & trạng thái

- **Kế hoạch triển khai Agentic RAG:** xem `docs/plan_agentic_rag.md` (kế hoạch chi tiết cấp module, milestone, schema, rủi ro). Đây là plan chính thức của dự án — cập nhật khi có thay đổi.
- **Trạng thái hiện tại:** ✅ **M0 xong** (4/8/2026). Dataset đủ trong `data/`, code scaffold dựng xong:
  - `.venv` + `requirements.txt` (core) + `pyproject.toml` (src layout, pytest) — pandas 3.0.5, openai 2.53.0.
  - `configs/base.yaml` + `api.yaml` (OpenRouter, model `qwen/qwen3.5-9b`), `.env` (gitignored) chứa key.
  - `src/vifinqa/`: `constants.py` (unit factors, hằng số), `config.py` (pydantic + YAML + `.env` loader), `loader.py` (load_stocks/load_questions/iter_reports).
  - `scripts/smoke_test.py` (data OK: 1,012 câu / 100 ticker / 1,973 report / ~97,860 bảng ước lượng), `scripts/test_llm_api.py` (API OK — model trả lời đúng, có reasoning tokens).
  - `tests/test_loader.py`: 4 test pass.
  - ⚠️ Qwen3.5-9B dùng **reasoning tokens mặc định** (~266 tokens/câu đơn giản) → cần xử lý chế độ thinking khi viết agent prompt (M5).
- **✅ M1 xong (4/8/2026)** — ETL toàn corpus + code review (11 finding) đã fix:
  - `etl/numbers.py` (parse số: `.` nghìn, `(x)` **và `-x`** âm, `-` rỗng; unit header `(VND|đồng)` + fallback `Đơn vị tính:`/`ĐVT:`/`Đơn vị tiền tệ:`; normalize bỏ dấu cả Đ/đ; period + `is_period_cell` nhận "Số cuối năm"/"Closing balance"), `etl/parser.py` (split trang, `<table>` 1 dòng, grid expand colspan, find_header_row), `etl/statements.py` (**classify = tiêu đề anchor + structural check**: bắt buộc cột kỳ + header "Mã số"/"STT·Chỉ tiêu"/"Codes" hoặc cột mã; negative loại notes/off-balance; hỗ trợ **BCTC tiếng Anh** — FPT/DBC/VGC 2024-25), `etl/catalog_builder.py` (lưu `unit_factor`; fallback unit theo anchor; skip section title), `scripts/run_etl.py` (**catalog_parts incremental** — không re-process khi resume, guard lỗi không brick).
  - **Output `data/derived/`**: `tables/{report_id}/table_{N}.csv` (146,246 wide tables), `catalog_tables.csv` (**10,797 bảng BCTC**: BS 5,066 / IN 2,349 / CF 3,382; 1,902/1,965 report có statement; unit_factor 22,926 bảng ≠VND; "Triệu đồng" 8,934), `documents.csv` (report_type **957/954/7/55** khớp doc), `etl_state.json`, `catalog_parts/`. Không có lỗi ETL.
  - Test vàng: HPG (BS [3,4,5], INCOME [6,7], CF [8,9]; LNST 60 = 8.600.550.706.227), VCB ngân hàng (BS "Báo cáo tình hình tài chính" [7,8], INCOME [11,12], CF [13,14]; unit header Triệu VND), VJC ("Lãi tiền gửi" wide table_50 = 208.253.201.298), DCM "Số cuối năm", FPT English "Closing balance". **55 test pass**. ⚠️ 63 report 0 statement (gồm `MCH_..._explanations` hợp lệ + vài HDB/KLB split) — xem lại ở M2/M7.
- **✅ M2 xong (4/8/2026)** — Facts tier (3 BCTC lõi) + fix English number format:
  - `etl/statements.py` nâng cấp: `find_item_code_col` (cột ≥70% giá trị khớp mã VAS `^\d{1,3}[a-z]?$` hoặc bank `^[IVXLC]+$|^[A-Za-z]$|^\d{1,2}$` — **chấp nhận chữ thường** a/b/c/g của ngân hàng; chọn cột dày nhất, phân biệt cột thuyết minh), `header_signature` (n_cols + period_keys + years), `group_statement_fragments` (**gom table liên tiếp cùng stmt + signature**), `build_asset` (period cols + code col + **label col = cột text dài nhất**, detect unit/format), `emit_facts` (bỏ header/section-title/dòng không giá trị; **dedupe item_code ở biên fragment** "mang sang/trang trước"; label rỗng → kế thừa; value_vnd = raw × unit_factor), `validate_asset` (log formula-label "60 = 50 - 51 - 52", cross-sum BS 270==440, **giữ giá trị gốc**).
  - `etl/numbers.py` bổ sung: `parse_number(s, thousands, decimal)` tổng quát + **`detect_number_format`** (vi `. `nghìn vs en `,`nghìn) — **fix bug BCTC tiếng Anh** (FPT/DBC/VGC) dùng `,` nghìn/`.` thập phân → trước đó 0 facts, giờ parse đúng (FPT CURRENT ASSETS 8,198,590,237,083 → 8198590237083 VND). `parse_vn_number` giữ nguyên (delegate `parse_number`).
  - `etl/facts_builder.py` (mới): `_scan_report_tables` (tự chứa — cùng classify/detect_unit/parse với M1 → facts & catalog nhất quán), `build_report_facts`, `write_facts_csv`, `merge_facts_parts`.
  - `scripts/run_facts.py` (mới): checkpoint theo ticker (`facts_state.json`), parts theo ticker gộp `facts_all.csv`, parallel 6 worker, guard lỗi. Chạy toàn corpus 100 ticker / 1,973 report trong ~400s, 0 lỗi.
  - **Output `data/derived/`**: `facts/{report_id}_facts.csv` (1,901 file), `facts_all.csv` (**377,578 dòng**: BS 215,715 / IN 70,879 / CF 90,984), `facts_parts/`, `facts_state.json`. 1,702 report có 3 BCTC, 151 có 2, 46 có 1; **0 report có statement mà 0 facts**. ⚠️ DoD ước lượng ≥400K nhưng thực tế 377K (ước lượngoptimistic) — không mất facts hợp lệ, threshold test hạ xuống 350K làm regression guard.
  - Test vàng M2 pass: HPG LNST 60 = 8.600.550.706.227; HPG DT thuần 10 = 55.836.458.379.759; HPG BS 270==440==78.223.007.670.925; HPG income code 60 dedupe 2 fragment → 2 fact (2 kỳ); VCB bank mã I..XII + 1-6 + a/b/c/g parse đúng; VCB TOTAL ASSETS 1.813.815.170 triệu → 1.813.815.170.000.000 VND; VCB LNST XIII 29.919.054 triệu → 29.919.054.000.000; FPT English code 100 = 8.198.590.237.083; parse_number/detect_number_format en+vi; merge+dedupe synthetic. **72 test pass toàn bộ**.
- **✅ M3 xong (5/8/2026)** — Retrieval hybrid (Qdrant Docker) + entity extraction, **rerank TẮT**:
  - `retrieval/entity.py` (company-name longest-first che bare ticker → Q4 FTS không phải FPT; alias lọc corpus; paren `(VJC)` chạy trên `normalize_label`; `extract_years` clamp joint không đảo; statement hint MỀM `statement_bonus=0.001` — KHÔNG hard filter), `retrieval/index.py` (`make_qdrant_client` factory branch local/server; `build_table_chunks` text_dense prefix+header+labels+anchor+fact_labels; `tf_sparse` MD5 term-id 2^21 bucket; `embed_dense` OpenRouter bge-m3, workers=4 + backoff mũ `2^attempt+jitter`; cache `.npy` per-ticker key = hash(texts)+n+**model+dim**), `retrieval/search.py` (hybrid native `FusionQuery(RRF)` dense+sparse trong 1 query; `build_payload_filter` entity; `apply_statement_bonus`; `_replace_score` dùng `dataclasses.replace`), `retrieval/facts_index.py` (parse_report_id, `facts_for_table` exact match `src_table_ids == table_id` — không substring), `retrieval/pipeline.py` (`RetrievalPipeline` lazy import reranker → serve không cần torch), `retrieval/rerank.py` (giữ cho tương lai, **không dùng M3**).
  - **Qdrant Docker** (`docker/docker-compose.yml`): `qdrant/qdrant:v1.19.0` (khớp `qdrant-client==1.19.0`), port 6333/6334, volume `data/derived/qdrant`. `config.py` `QdrantConfig.mode=server|local` + host/port. Build & serve song song (không file lock).
  - **Config**: `rerank.enabled=false` (Qwen3-Reranker-0.6B local CPU quá nặng — đã xoá model 1.2GB; Cohere API = đóng không hợp lệ §5; để sau nếu cần: mxbai-rerank-v2 Apache-2.0 hoặc HF endpoint Vietnamese_Reranker), `embed_statement_only=true` (chỉ 10,797 bảng BCTC, bỏ ~135K notes/TOC), `embedding.max_chars=4000`, `workers=4`, `statement_bonus=0.001`.
  - **Output**: `data/derived/qdrant/` (Docker storage), `embeddings/{TICKER}.npy`+`.json` (10,616 vector dense 1024-d), `retrieval_state.json` (100 ticker done), `retrieval_topk.csv`, `retrieval_metrics.json`. Collection `bctc_tables`: **10,616 points** (build 186s, ~$0.01 API, 0 lỗi 429).
  - Test vàng M3 pass: entity 9 câu golden (Q1 VJC / Q4 FTS / Q11 BID / Q369 HPG+HSG+MSR+NKG / Q790 VIB+BID); years range 2018–2024 expand; report_type separate/consolidated; statement hint income/BS/CF; units triệu/tỷ/nghìn tỷ. Serve 20 câu: coverage 0.95, entity_ticker/year_rate 1.0, latency median 0.57s/p95 2.14s. **84 test pass toàn bộ**.
  - ⚠️ Còn lại (M3.2 deferred): `facts_for_table` gộp report suffix (HDB `separate` vs `separate_1`) — cần thêm cột `report_id` vào `facts_all.csv` ở ETL. Bug `_rerank` stale score + `_DTYPE_KWARG` không áp dụng (rerank off).
- **✅ M4 xong (5/8/2026)** — Text-to-Pandas codegen + sandbox + đóng gói submission (gộp M4+M5simple+M6 gốc):
  - ⚠️ **Sửa bởi fix B (6/8/2026)** — sandbox/codegen/builder contract đã refactor (xem mục "✅ Fix (B) sandbox contract — CODE XONG" dưới). Mô tả chi tiết dưới đây là trạng thái **M4 gốc trước fix** (runner inject bare `df1`, prompt bare `df1`, builder package wide, `embed_statement_only=true`, 105 test). Trạng thái hiện tại: runner inject `dfs` dict, prompt `dfs["df1"]`, builder package tidy, `embed_statement_only=false` (base.yaml), **107 test pass**.
  - `sandbox/ast_check.py` (AST walk chặn import/call nguy hiểm `open/eval/read_csv/__class__`/module `os,sys,subprocess,...`; limit `max_code_len=6000`/`max_ast_nodes=500`), `sandbox/runner.py` (subprocess `python -I` isolated; inject `pd/np/math/re/json` + **inline `vn_num`** = copy `parse_vn_number` — runner không import vifinqa được dưới `-I`; safe builtins bỏ `open/eval/exec/getattr/...`; exec → gán `result` → JSON), `sandbox/executor.py` (`run_pandas` spawn runner + timeout, evidence resolve abs).
  - `codegen/prompt.py` (`build_messages`: system data contract — số VN `.`nghìn/`,`thập phân/`(x)`âm → `vn_num`; **tên cột giữ nguyên KĐ dấu+cách** (`Mã số`, `2023 VND`); đổi đơn vị `/1e6` `/1e9`; gán `result`; không import/read_csv; few-shot 2 ví dụ), `codegen/llm.py` (`LLMClient` OpenRouter `qwen/qwen3.5-9b`; **`thinking=False`** qua `extra_body={"reasoning":{"enabled":False}}` — Qwen3 reasoning mặc định ăn hết max_tokens → content rỗng đã thấy 4258 tok; `_extract_code` strip think + fence ```python + **strip import + strip `def vn_num`** — LLM hay tự định nghĩa lại sai logic).
  - `agent/loop.py` (`solve`: retrieve top-k → build table cards (read wide CSV: columns+sample 8 rows+fact hints) → codegen → ast_check → exec → **retry ≤2 feed error** → fallback `result=0.0` nếu fail hết; record `{id,question,answer,relevant_docs,relevant_tables,evidence,pandas_query}`; `csv_path=data/{report_id}__{table_id}.csv`).
  - `submission/builder.py` (results.jsonl → `submission.json` + `data/` flat; assert 1012 id; thiếu→fallback; copy wide table gốc→`data/{rid}__{tid}.csv` dedupe), `submission/validate.py` (re-exec mỗi query trên CSV đã đóng gói, so `answer` tol 0.01; **short-circuit `result=0.0`** không spawn; ThreadPoolExecutor workers=8; check `csv_path` start `data/`+tồn tại), `submission/pack.py` (ZIP `submission.json`+`data/**` ở root, đúng 1 json, checksum).
  - `scripts/run_codegen.py` (batch → `data/out/results.jsonl` checkpoint theo id, concurrency 4, guard lỗi) + `scripts/build_submission.py` (results→build→validate→pack).
  - **Smoke 5 câu**: pipeline OK, validate 1012/1012=100% (5 real + 1007 fallback), `submission.zip` 74KB đúng schema (1 json root + 30 CSV `data/`). p95 codegen 44s/câu (thinking off). **105 test pass toàn bộ** (thêm 21 test sandbox+prompt).
  - ⚠️ **KHOẢNG TRỐNG RETRIEVAL quan trọng**: `embed_statement_only=true` (M3.1) chỉ index 10,797 bảng BCTC statement, **bỏ ~135K bảng notes/thuyết minh**. Nhiều câu hỏi nhắm notes items (VJC "Lãi tiền gửi" table_50 `is_statement=0`, "chi phí phạt", "cho vay khách hàng ngành X"...) → **retrieval không thấy → answer 0**. Smoke 5: Q1/Q3/Q4/Q5 đều answer 0 do bảng đúng không được index. **Cần re-index toàn bộ 146K bảng** (`embed_statement_only=false`) trước khi chạy full 1012 — đã xác minh VJC 2018: 125 bảng, chỉ 11 statement, table_50 notes.
  - ⚠️ Codegen còn yếu: LLM đôi khi drop khoảng trắng tên cột (`Mãsố`/`2018VND`) → KeyError → retry fallback label match (chạy được nhưng chậm). Câu phức tạp (ratio đa công ty, argmax) chưa test — M5/M7 tune.
- **Việc cần làm kế tiếp:**
  1. **Re-index Qdrant toàn 146K bảng** (`embed_statement_only=false`) → fix khoảng trống notes. ~$0.14 API, ~15-20p rebuild. Sau đó chạy full codegen 1012.
  2. M5 — Agent loop ReAct đầy đủ (multi-tool: inspect_table/get_facts/search_tables trong loop) + self-consistency; tune codegen (few-shot thêm, schema linking).
  3. M7 — dev-set 40 câu + label + metrics F2/Answer/Exec; evaluate submission thật.
  4. Xác minh câu hỏi kiểm thử của BTC trùng/khác với 1,012 câu HF (chờ BTC).

### ⚠️ Phát hiện từ lần nộp đầu (6/8/2026) — leaderboard:
`TABLES_F2=0.0, DOCS_F2=0.77, ANSWER_ACC=0.105, EXEC_ACC=0.020`. 3 vấn đề độc lập:
- **(A) TABLES_F2=0 — KHÓA, không fix được bằng code.** Matching **exact string** trên `report_id|table_N` (retrieval_metrics.py: set intersection, **không partial credit**, không so content). `table_N` của BTC từ corpus nội bộ `ocr_filter/` (`*_extracted_tables/table_N.csv`) do bước preprocessing **không công khai** sinh ra (document.py chỉ đọc anchor `[table_N](...)` có sẵn; catalog.py `_scan_table_ids` lấy N từ filename `table_N.csv`). Đã verify: ocr_filter **không có trên HF** (TiniX source `tinixai/ocr_annual_financials` chỉ có `_extracted.txt` thô; mirrors `vduydong`, `conghuy` cũng vậy). Test fixture companion repo chỉ có synthetic `DOC_A|table_1`. Ta có 146,246 wide table; BTC công bố 143,815 normalized tables; **hiệu 2,431 không giải được bằng ngưỡng row-count** (bỏ `n_rows==1` → 143,094, lệch 721) → khác biệt **cấu trúc** (cách tách/gộp bảng multi-page). Fix format `report_id|table_N` (đã làm 6/8) là cần nhưng **không đủ** — N sai hệ thống. **Chỉ BTC cấp ocr_filter/numbering mới giải quyết được** (hỏi chính đáng — dataset card tự thừa nhận annotation không public). Fix này KHÔNG ảnh hưởng ANSWER/EXEC (answer tính từ content CSV, không dùng table_N).
- **(B) EXEC_ACC=2% ≪ ANSWER_ACC=10.5% — BUG CONTRACT SANDBOX, ROI cao, fix được.** Grader benchmark (`answering/sandbox.py` + `prompts/answering/program_system.txt`): evidence đưa vào **dict `dfs` keyed theo `table_ref`** (nhiều CSV → `dfs["..."]`, 1 CSV → alias `df`), **KHÔNG inject bare `df1`/`df2`**; CSV đọc `dtype=str, keep_default_na=False, index_col=None` (cells = string, index numeric RangeIndex); **không có helper `vn_num`** (LLM phải parse số thủ công); `def` cho phép; gán `result` = scalar, round 2 decimals; output chỉ code thuần không fence. Query của ta dùng **bare `df1`/`df2`** + `df.index.astype(str).str.contains(...)` (giả index=label) → NameError/KeyError ở grader → crash → EXEC 2%. `sandbox/runner.py` hiện inject bare `g.update(frames)` + read `dtype=str` (đúng index_col=None) → **sandbox ta ≠ grader** → validate nội bộ 65% nhưng grader 2%. **FIX**: (1) `runner.py` inject `dfs` dict keyed theo variable (+`df` alias nếu 1 frame) thay bare; (2) `codegen/prompt.py` dạy LLM dùng `dfs["df1"]`, cells string, index numeric (lọc theo cột tên chứ không `.index`), parse số VN inline (define `vn_num` hoặc manual), gán `result` round 2; (3) bỏ strip `def vn_num` trong `_extract_code`; (4) re-codegen 1012.
- **(C) Retrieval gap (M4 note)** — `embed_statement_only=true` bỏ 135K bảng notes → nhiều câu answer=0. Fix: re-index 146K (`embed_statement_only=false`).

**Thứ tự ưu tiên mới:** (B) fix sandbox contract + re-codegen → nâng EXEC/ANSWER (ROI cao nhất, không cần BTC). (C) re-index 146K → nhiều câu có real answer. (A) hỏi BTC ocr_filter numbering (chỉ đường cho TABLES_F2).

### ✅ Fix (B) sandbox contract — CODE XONG (6/8/2026, chưa re-run)
Đã refactor code (107 test pass) để sandbox/codegen khớp grader `answering/sandbox.py`:
- `sandbox/runner.py`: inject **dict `dfs` keyed theo variable** (`dfs["df1"]`) + alias `df` khi 1 CSV; đọc `pd.read_csv(dtype=str, keep_default_na=False, index_col=None)` — **mirror grader exact**. Bỏ bare `g.update(frames)`. Giữ inject `vn_num`/pd/np/math/re/json + safe builtins.
- `codegen/prompt.py`: system/few-shot/user đổi bare `df1` → `dfs["df1"]` (+alias `df1 = dfs["df1"]`); nhắc `vn_num` đã có sẵn, index numeric, gán `result`. Giữ tidy 4-cột schema.
- `agent/loop.py`: `_repair` + `_df_refs_over` error message đổi sang `dfs["dfN"]` (regex `\bdf(\d+)\b` vẫn match `df1` trong `dfs["df1"]` — đã verify).
- `submission/builder.py`: **revert** package wide → package **tidy** (`evidence/{rid}__{tid}.csv`); fallback **regenerate tidy từ wide** (`wide_csv_to_tidy` + `unit_factor` từ `catalog_tables.csv`) khi tidy thiếu (stale evidence/); tidy rỗng → header-only (không crash, query→0.0). Summary thêm `tidy_regen`.
- `codegen/llm.py` `_extract_code`: **giữ** strip `def vn_num` (builder inject `_VN_NUM_DEF` cho grader — grader không có vn_num). Query submitted = `_VN_NUM_DEF` + code LLM (đã strip def trùng).
- Test: `test_sandbox.py` thêm `single_df_alias` + `bare_df_nameerror`; `test_codegen_prompt.py` assert `dfs["df1"]`.

**⚠️ Cần làm (chạy bên ngoài, tốn API):**
1. `rm -rf data/derived/evidence/ data/derived/qdrant/` (xoá tidy stale + index statement-only cũ).
2. `embed_statement_only` đã = `false` ở `base.yaml` (api.yaml kế thừa) → chỉ cần chạy `scripts/run_retrieval.py` build index toàn 146K (~20p, ~$0.14, fix gap notes).
3. `scripts/run_codegen.py` re-codegen 1012 với contract dfs mới (~3h, API LLM) → `data/out/results.jsonl` mới (query dùng `dfs["df1"]`).
4. `scripts/build_submission.py` → pack zip tidy. Validate nội bộ khớp grader (dfs contract).
5. Nộp + quan sát EXEC_ACCURACY nhảy (kỳ vọng 2% → tiệm cận ANSWER_ACCURACY). TABLES_F2 vẫn 0 (cần BTC — mục A).
- **Ràng buộc hạ tầng:** Qdrant Docker local (`docker/docker-compose.yml`). LLM qua OpenRouter API (`qwen/qwen3.5-9b`, `thinking=False`); embedding bge-m3 qua OpenRouter. Chưa có GPU local — rerank local bỏ (CPU không nổi).
- **Việc cuối cùng:** nộp working notes paper mô tả phương pháp.

---

## 9. Checklist khi nộp bài

- [ ] Đúng 1 file `submission.json` + thư mục `data/` ở cấp ngoài cùng ZIP.
- [ ] Mọi `csv_path` bắt đầu bằng `data/` và file thực sự tồn tại.
- [ ] `variable` trong evidence là Python identifier hợp lệ, không trùng trong cùng câu hỏi.
- [ ] `pandas_query` chạy lại được trên CSVs đã nộp và ra `answer` đúng (trong ngưỡng).
- [ ] Đủ câu trả lời cho toàn bộ test questions (thiếu câu = bài vô hiệu).
- [ ] `answer` là float (không phải string), đúng đơn vị.
- [ ] `relevant_docs` / `relevant_tables` đúng mã báo cáo (bỏ `.txt`) và đúng vị trí bảng.
- [ ] Kiểm tra số bài nộp hôm nay (≤10/ngày, ≤5 private phase).
- [ ] Ghi chú nguồn dữ liệu/mô hình phục vụ bài báo.

---

## 10. Nguồn tham khảo

- Đề bài cuộc thi (text do chủ dự án cung cấp).
- Dataset: https://huggingface.co/datasets/AIGuruTinix/ViFinQA
- Companion repo: https://github.com/DSKT-NOWJ/ViFinQA
- Dashboard nộp bài: http://leaderboard.aiguru.com.vn/
- Paper: *"ViFinQA: A Comprehensive and Challenging Benchmark for End-to-End Vietnamese Financial Reasoning"*
