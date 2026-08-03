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
- **`relevant_tables`:** định dạng `<id_báo_cáo>|<vị trí bảng>`. Vị trí bảng = số thứ tự/vị trí bảng trong báo cáo **theo dữ liệu BTC cung cấp** (ví dụ `|350`). ⚠️ Cần xác minh quy ước đánh số bảng của BTC (có thể trùng với `table_N` trong pipeline repo).
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
- **LLM code/pandas:** Qwen2.5-Coder-14B-Instruct, Qwen3-14B / Qwen3-8B (phát hành 4/2025 ✅), DeepSeek-R1-Distill-Qwen-14B / Distill-Llama-8B (reasoning), Phi-4 (14B, 2/2025 ✅), Gemma-2-9B, Llama-3.1-8B.
- **Embeddings:** BAAI/bge-m3 (~568M ✅), Qwen3-Embedding-0.6B/4B/8B ✅ (không vượt 14B).
- **Reranker:** bge-reranker-v2-m3 (~568M ✅), Qwen3-Reranker-0.6B/4B/8B ✅.
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
[3] Retrieval: lọc entity → hybrid BM25 + dense (BGE-M3) → RRF → rerank (BGE-reranker-v2-m3) → top-k bảng
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

- **Trạng thái hiện tại:** Mới bắt đầu. Thư mục `D:\GURU` trống. Chưa có dữ liệu tải về, chưa có code.
- **Việc cần làm kế tiếp:**
  1. Tải dataset ViFinQA từ Hugging Face (questions, code_stock.csv, financial_statements).
  2. Xác minh câu hỏi kiểm thử của BTC trùng/khác với 1,012 câu HF.
  3. Khảo sát 2–3 file .txt OCR để nắm cấu trúc thật (markup bảng, marker trang, header, đơn vị).
  4. Quyết định tech stack (Python, thư viện pandas/polars, framework LLM local — vLLM/transformers), kiểm tra phần cứng (GPU?) so với model 14B.
  5. Dựng ETL v1 cho 1 công ty mẫu, benchmark thủ công vài câu hỏi.
- **Ràng buộc hạ tầng:** (chưa xác định — cần cập nhật: có GPU/VRAM bao nhiêu? chạy local hay server?)
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
