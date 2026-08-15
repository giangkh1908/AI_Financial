# GURU — ViFinQA: Trả lời câu hỏi tài chính từ báo cáo tài chính Việt Nam

Pipeline tổng quát: **OCR BCTC (HTML) → bảng chuẩn hoá (canonical) → facts/evidence → retrieval (Qdrant) → codegen (LLM) → submission**.

Mục tiêu: từ **1012 câu hỏi** (tiếng Việt, đơn vị nghìn/triệu/tỷ đồng) truy vấn số liệu chính xác trong **~1965 báo cáo tài chính** của **100 doanh nghiệp Việt Nam** (2015–2025), trả `pandas_query` + đáp án số.

Tài liệu này mô tả **toàn bộ định dạng dữ liệu** qua các tầng ETL — mỗi tầng có **ví dụ thật** lấy từ corpus (ACB, VJC, HT1, HHV...) để bạn đọc hiểu mà không cần mở file.

---

## Mục lục
1. [Đề bài & dữ liệu gốc](#1-đề-bài--dữ-liệu-gốc-datad)
2. [Ví dụ xuyên suốt: 1 câu hỏi đi qua toàn pipeline](#2-ví-dụ-xuyên-suốt-1-câu-hỏi-đi-qua-toàn-pipeline)
3. [Định dạng dữ liệu — từng tầng](#3-định-dạng-dữ-liệu--từng-tầng)
4. [Pipeline ETL (canonical wide làm nguồn sự thật)](#4-pipeline-etl-canonical-wide-làm-nguồn-sự-thật)
5. [Chunking & Embedding](#5-chunking--embedding)
6. [Agent & codegen](#6-agent--codegen)
7. [Vấn đề đang gặp phải](#7-vấn-đề-đang-gặp-phải)
8. [Cách chạy](#8-cách-chạy)
9. [Cấu trúc repo](#9-cấu-trúc-repo)

---

## 1. Đề bài & dữ liệu gốc (`data/`)

### 1.1. `data/questions/questions.jsonl` — 1012 câu hỏi

1 dòng/1 JSON: `{"id": 1, "question": "Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC) là bao nhiêu triệu đồng?"}`

**Phân bố loại bảng cần truy vấn** (thống kê toàn bộ 1012 câu):

| Loại | Số câu | Ví dụ |
|---|---|---|
| Notes/thuyết minh/segment | **437** | "Số dư cho vay ngành Thương mại ACB" |
| Income (KQKD) | 289 | "Lợi nhuận sau thuế FTS 2023" |
| Balance sheet (CĐKT) | 238 | "Vốn chủ sở hữu FIT" |
| Cash flow (LCTT) | 48 | "LCTT thuần HĐKD VSC 2017" |

**Đơn vị hỏi linh hoạt**: nghìn tỷ / trăm tỷ / tỷ / triệu / nghìn / đồng — thường **khác đơn vị lưu** → pipeline phải quy đổi về VND rồi chia ra đơn vị câu hỏi.

**Cấu trúc câu hỏi điển hình** (entity cần trích xuất):
- Ticker: `VJC`, `ACB`, `CTCP Hàng không Vietjet (VJC)`, alias `Vietjet`...
- Năm: `2018`, `31/12/2022`, `cuối năm 2022`, range `2018–2024`...
- Loại báo cáo: `công ty mẹ` → separate; `hợp nhất` → consolidated; không nhắc → cả hai.
- Chỉ tiêu: `lãi tiền gửi`, `chi phí lương`, `thù lao HĐQT`...
- Đơn vị hỏi: `triệu đồng`, `tỷ đồng`, `phần trăm`...

### 1.2. `data/code_stock.csv` — 100 mã chứng khoán

Cột: `Mã CK, Tên công ty`. Ví dụ:
```
VJC,CTCP Hàng không Vietjet
ACB,Ngân hàng TMCP Á Châu
```
Dùng cho entity extraction (match tên công ty → ticker).

### 1.3. `data/financial_statements/{TICKER}/{YEAR}/{report_id}/`

File `{report_id}_extracted.txt` = **OCR thô** của BCTC, mỗi trang đánh dấu `===== PAGE N =====`, các `<table>` HTML **nằm gọn trong 1 dòng**.

`report_id` = `{TICKER}_financial_statements_{YEAR}_{type}` với `type ∈ consolidated|separate|aggregated|other`. Một số report bị tách phần → hậu tố `_1`, `_2` (vd `VPB_financial_statements_2022_separate_1`).

**Ví dụ OCR thật — ACB 2022 hợp nhất:**
```
===== PAGE 1 =====
KPMG
Ngân hàng Thương mại Cổ phần Á Châu
Báo cáo tài chính hợp nhất cho năm tài chính kết thúc ngày 31 tháng 12 năm 2022

===== PAGE 2 =====
<table><tr><td rowspan="2">Giấy phép hoạt động Ngân hàng</td><td colspan="2">Số 91/GP-NHNN ngày 19 tháng 9 năm 2018</td></tr>...
```

⚠️ Cell dùng thuộc tính HTML `rowspan`/`colspan` (header ngân hàng hay `rowspan=2 colspan=2`) → **bắt buộc expand khi parse** (đây từng là bug lệch cột — xem mục 7).

---

## 2. Ví dụ xuyên suốt: 1 câu hỏi đi qua toàn pipeline

**Câu Q1:** *"Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC) là bao nhiêu triệu đồng?"*

| Bước | Input | Output thật |
|---|---|---|
| ① Entity | câu hỏi | `ticker=VJC, year=2018, report_type=separate, statement=None, unit_factor=1e6 (triệu)` |
| ② Retrieve | câu hỏi → embed → Qdrant | top-k bảng, rank #1 = `VJC_financial_statements_2018_separate\|table_50` |
| ③ Chunk | catalog dòng table_50 | `text_dense` 526 ký tự (xem mục 5) |
| ④ Embed | text_dense → BGE-M3 (bge_m3_server) | vector 1024-dim, chuẩn hoá norm=1 |
| ⑤ Evidence | wide → tidy | `[lai tien gui, , 2018, 208253201298.0]` |
| ⑥ Answer | deterministic match label | **208253.2 triệu đồng** |

---

## 3. Định dạng dữ liệu — từng tầng

### Tổng quan 7 tầng

```
OCR gốc                    T0: {rid}_extracted.txt      (trang + <table> HTML)
   → wide CSV              T1: tables/{rid}/table_N.csv (grid rowspan-aligned)
   → metadata              T2: catalog_tables.csv + layouts/{rid}.json
   → tidy evidence         T3: evidence/{rid}__table_N.csv  [chi_tieu, Mãsố, ky, value]
   → facts (3 BCTC lõi)    T4: facts_all.csv               (long-format VND)
   → merged statement      T5: evidence_merged + statement_meta.csv
   → embed cache           T6: embeddings/{ticker}.npy
```

### Tầng 0 — OCR gốc
Xem ví dụ mục 1.3. Đây là **đầu vào duy nhất** của ETL, không chỉnh sửa.

### Tầng 1 — CANONICAL WIDE `tables/{rid}/table_{N}.csv`

Grid 2D đã **expand colspan + rowspan** + entity-decoded (BeautifulSoup). `N` = số thứ tự `<table>` trong report (1-based, không reset theo trang). Giá trị giữ **chuỗi OCR gốc** — chưa parse số.

**Ví dụ thật — ACB 2022 `table_2` (CĐKT ngân hàng, 5 cột):**
```
0  [''      , ''                       , 'Thuyết minh', 'Tại ngày'          , 'Tại ngày']
1  [''      , ''                       , 'Thuyết minh', '31.12.2022 Triệu VND', '31.12.2021 Triệu VND']
2  ['A'     , 'TÀI SẢN'                , ''            , ''                 , '']
3  ['I'     , 'Tiền mặt, vàng bạc, đá quý', '4'         , '8.460.892'        , '7.509.877']
4  ['II'    , 'Tiền gửi tại Ngân hàng Nhà nước', '5'    , '13.657.531'       , '32.349.574']
```
- Hàng 0-1: header 2 tầng (dòng marker + dòng năm/đơn vị).
- Cột: `code`(A/I/II) | `label` | `thuyết minh` | kỳ 2022 | kỳ 2021.
- Số `8.460.892` = 8.460.892 **triệu** VND (đơn vị trong header) → value = 8.460.892 × 1e6.

### Tầng 2 — `catalog_tables.csv` (metadata mọi bảng)

1 dòng/bảng, 23 cột. **Ví dụ dòng `table_2` ACB 2022:**
```
report_id: ACB_financial_statements_2022_consolidated
ticker:    ACB          year: 2022        report_type: consolidated
table_id:  table_2      page_no: 7        unit: Triệu VND        unit_factor: 1000000.0
is_statement: 1         statement: balance_sheet
header_text:  |  | Thuyết minh | 31.12.2022 Triệu VND | 31.12.2021 Triệu VND
row_labels: TÀI SẢN | Tiền mặt, vàng bạc, đá quý | Tiền gửi tại Ngân hàng Nhà nước | ...
n_rows: 40              n_cols: 5
anchor_context: NGÂN HÀNG THƯƠNG MẠI CỔ PHẦN Á CHÂU\nMẫu B02/TCTD-HN\nBÁO CÁO TÌNH HÌNH TÀI CHÍNH HỢP NHẤT
start_line: 164        # dòng vật lý của <table> trong OCR → relevant_tables submission
header_idx: 1          period_row_idx: 1
code_col: 0            label_col: 1
period_cols: [3, 4]    thuyet_minh_cols: [2]     number_format: vi
```

| Cột | Ý nghĩa |
|---|---|
| `report_id` / `ticker` / `year` / `report_type` | Nhận diện báo cáo (consolidated/separate/aggregated/other) |
| `table_id` | `table_{N}` — định danh bảng trong report |
| `page_no` | Trang OCR chứa bảng |
| `unit` + `unit_factor` | Đơn vị hiển thị + hệ số × VND (Triệu VND → 1e6) |
| `is_statement` + `statement` | `1` + `balance_sheet\|income\|cash_flow` nếu BCTC lõi; `0` + rỗng nếu notes |
| `header_text` | Dòng header của bảng (tên cột) |
| `row_labels` | Tên các dòng/chỉ tiêu — **chỉ 10 dòng đầu** ⚠️ (Vấn đề #1) |
| `n_rows` / `n_cols` | Kích thước grid |
| `anchor_context` | 6 dòng text trước bảng trong OCR (context phân loại) |
| `start_line` | Vị trí dòng bảng trong OCR → format `report_id\|start_line` khi nộp |
| `header_idx` / `period_row_idx` | Dòng header / dòng nhãn kỳ |
| `code_col` / `label_col` | Cột mã số / cột tên chỉ tiêu |
| `period_cols` | Cột chứa giá trị kỳ (vd `[3, 4]`) — JSON |
| `thuyet_minh_cols` | Cột "Thuyết minh" (bỏ qua khi emit facts) |
| `number_format` | `vi` (`.`,=nghìn) hay `en` (`,`,=nghìn) |

**Bảng đại diện 6 loại** (header + row_labels thật):

| Loại | Ví dụ | header_text | row_labels (thật, rút gọn) | period_cols |
|---|---|---|---|---|
| Statement CĐKT | ACB `table_2` | `Mã số\|NGUỒN VỐN\|Thuyết minh\|Số cuối năm\|Số đầu năm` | `A. TÀI SẢN \| I. Tiền mặt... \| II. Tiền gửi NHNN...` | [3,4] |
| Notes thu nhập | VJC `table_50` | `\|2018VND\|2017VND` | `Lãi tiền gửi \| Lãi chênh lệch tỷ giá... \| Lãi từ thanh lý CTy con` | [0,1] |
| Segment ngành | ACB `table_35` | `\|31.12.2022Triệu VND\|31.12.2021Triệu VND` | `Thương mại \| Sản xuất và gia công chế biến \| Xây dựng \| Dịch vụ...` | [1,2] |
| % không cột năm | HHV `table_1` (Q19) | `Tên Công ty\|Tỷ lệ lợi ích\|Tỷ lệ biểu quyết\|Vốn đầu tư (VND)` | `Cty BOT Bắc Giang \| Cty Đầu tư Đèo Cả \| Cty Phước Tượng...` | **rỗng** ⚠️ |
| Notes thù lao | MPC `table_61` | `\|2021VND\|2020VND` | `Chu Thị Bình \| Lê Văn Quang \| Lê Văn Điệp \| Bùi Anh Dũng...` | [0,1] |
| Notes chi phí | FTS `table_45` | `STT\|Loại chi phí quản lý CTCK\|Năm nay\|Năm trước` | `Chi phí lương... \| BHXH, BHYT... \| Chi phí đào tạo...` | [2,3] |

### Tầng 2b — `layouts/{rid}.json` (layout chi tiết 1 bảng)

JSON `{table_id: {...}}` cho **mọi bảng** (kể cả notes/segment). Ví dụ `table_2`:
```json
{
  "header_idx": 1, "period_row_idx": 1,
  "code_col": 0, "label_col": 1,
  "period_cols": [3, 4], "thuyet_minh_cols": [2],
  "unit_factor": 1000000.0, "unit_label": "Triệu VND",
  "number_format": "vi"
}
```

Notes/segment cũng có layout — vd ACB `table_37` (cho vay theo ngành): `label_col=0, period_cols=[1,2], unit_factor=1e6`. → tidy ra đúng `value`.

### Tầng 3 — `evidence/{rid}__{table_N}.csv` (tidy mọi bảng, 4 cột)

**Schema grader contract** — mọi bảng (statement + notes) ép về `[chi_tieu, Mãsố, ky, value]`:

| chi_tieu | Mãsố | ky | value |
|---|---|---|---|
| `tien mat, vang bac, da quy` | `I` | 2022 | 8460892000000.0 |
| `tien mat, vang bac, da quy` | `I` | 2021 | 7509877000000.0 |
| `lai tien gui` (VJC table_50) | | 2018 | 208253201298.0 |
| `thuong mai` (ACB table_37) | | 2022 | 73260878000000.0 |

- `chi_tieu`: label **chuẩn hoá ASCII** (bỏ dấu, hạ thường) — `Tiền mặt...` → `tien mat, vang bac, da quy`.
- `Mãsố`: mã VAS/bank (`I`, `322`, `411a`) — rỗng nếu notes không có.
- `ky`: năm (`2022`); **bỏ số dư đầu kỳ** (01/01/2022) để tránh nhập nhằng cùng năm.
- `value`: **VND sạch** = `vn_num(cell) × unit_factor` (8.460.892 triệu → 8.460.892.000.000 VND).

### Tầng 4 — `facts/{rid}_facts.csv` + `facts_all.csv` (3 BCTC lõi, long-format)

Chỉ **balance_sheet/income/cash_flow**, 12 cột. Ví dụ:
```
ticker ACB | year 2022 | report_type consolidated | statement balance_sheet
item_code I | item_label "tien mat, vang bac, da quy" | item_label_raw "Tiền mặt, vàng bạc, đá quý"
period_key flow_year | period_label "31.12.2022 Triệu VND" | value_vnd 8460892000000.0 | src_table_ids table_2
```
- `item_label` = ASCII chuẩn; `item_label_raw` = gốc có dấu (diagnostics).
- `value_vnd` = chuẩn VND tuyệt đối.
- `src_table_ids` = bảng vật lý gốc (fix fragment-split khi CF 2 trang).

### Tầng 5 — `evidence_merged/{rid}__{stmt}.csv` + `statement_meta.csv`

Statement **gộp toàn bộ chỉ tiêu** (fix bảng bị tách nhiều fragment → 1 bảng):
- `evidence_merged/ACB_...__balance_sheet.csv` — giống tidy nhưng đủ hết chỉ tiêu của CĐKT.
- `statement_meta.csv`: map bảng vật lý → statement:
```
report_id | statement     | src_table_ids
AAA_2015_consolidated | balance_sheet | ["table_2", "table_3"]
AAA_2015_consolidated | income        | ["table_4"]
```

### Tầng 6 — `embeddings/` (cache embed per ticker)

`{ticker}.npy` (float32 `[N, 1024]`) + `{ticker}.json` `{hash, n, model, dim}` — hash md5 của text chunks → **resume không embed lại** (cache-hit khi text không đổi). Cache key gồm `model`+`dim` → đổi embed model/dim (vd nemotron 2048 → bge-m3 1024) tự invalidate, không cần xoá tay (chạy `--rebuild` để drop collection + xoá checkpoint).

### `documents.csv`

1 dòng/report: `report_id, ticker, year, report_type, company_name, has_consolidated, has_separate`.

---

## 4. Pipeline ETL (canonical wide làm nguồn sự thật)

```
OCR txt → parse_table_grid (rowspan) → format_classify (TableLayout)
    │
    ▼
wide CSV  tables/{rid}/table_N.csv        ← CANONICAL (1 bản, không nhân đôi)
    + layouts/{rid}.json + catalog_tables.csv
    │
    ├─▶ facts_all.csv        (3 BCTC lõi → long VND)
    ├─▶ evidence_merged/     (statement gộp, fix fragment-split)
    ├─▶ evidence/            (tidy mọi bảng, kể cả notes)
    └─▶ index → Qdrant       (chunks + embed)
```

**Nguyên tắc:** mọi consumer (facts/tidy/merged/index/codegen) đọc `TableLayout` từ catalog/layouts — **cấm tự đoán cột lại**. Bug lịch sử (lệch cột bank header) do mỗi nơi re-heuristic → đã fix bằng 1 nguồn sự thật.

### `src/vifinqa/etl/parser.py`
- `split_pages`: tách trang theo `===== PAGE N =====`.
- `parse_table_grid`: HTML `<table>` → grid 2D, **expand colspan + rowspan** (fix header ngân hàng `rowspan=2 colspan=2`), entity-decoded qua BeautifulSoup (`&#x27;` → `'`).

### `src/vifinqa/etl/format_classify.py`
Mọi bảng → `TableLayout(header_idx, period_row_idx, code_col, label_col, period_cols, thuyet_minh_cols, unit_factor, unit_label, number_format)`:
- Header thật = dòng marker "Mã số"/"Chỉ tiêu"/"STT"/"Codes"; period = cột có năm/ngày.
- Header 2 tầng (ngân hàng): nhãn kỳ ở dòng ngay sau header → `period_row_idx = header_idx + 1`.
- Statement (balance_sheet/income/cash_flow) qua `classify_statement` (anchor + cấu trúc); **notes/segment** cũng detect layout (period cols + unit).
- MSR 2015-2018: bảng chứa cả cột "Tập đoàn" + "Công ty" → lọc theo `report_type`.

### `src/vifinqa/etl/catalog_builder.py`
- Viết wide CSV + catalog (23 cột) + `layouts/{rid}.json`.
- `start_line` = dòng vật lý bảng trong OCR → `relevant_tables` submission.
- `header_and_labels` lấy `row_labels` = **10 dòng đầu** cột `label_col` (⚠️ Vấn đề #1).

### `src/vifinqa/etl/tidy.py` — evidence 4 cột
`grid_to_tidy` dùng layout (label_col/code_col/period_cols/unit_factor, period_row_idx cho header 2 tầng) — không đoán cột. `value = vn_num(cell) × unit_factor`. ⚠️ Bảng không có `period_cols` (vd bảng %) → **không sinh row** (Vấn đề #5).

### `src/vifinqa/etl/facts_builder.py` + `merged_evidence.py`
3 BCTC lõi → facts long-format → gộp statement `[chi_tieu, Mãsố, ky, value]`, fix fragment-split, bỏ số dư đầu kỳ.

---

## 5. Chunking & Embedding

### 5.1. Đơn vị chunk = 1 bảng

Mỗi `<table>` trong OCR = **1 chunk** = 1 vector trong Qdrant (118,728 bảng = 118,728 points). Không tách theo đoạn văn.

### 5.2. Cấu trúc text chunk (`build_table_chunks`)

Ghép 4-5 phần nối bằng `\n`, cắt ở `max_chars`:
```python
prefix = "BÁO CÁO {report_type} {year} | {statement} | {ticker}"
header = "header_text"      # tên cột bảng
labels = "row_labels"       # tên các dòng/chỉ tiêu (10 dòng đầu)
anchor = "anchor_context"   # 6 dòng text trước bảng trong OCR
fact_labels = item_label từ facts (chỉ BCTC lõi)
```

**Ví dụ chunk dense thật — VJC `table_50` (526 ký tự):**
```
BÁO CÁO separate 2018 | nan | VJC
| 2018VND | 2017VND
Lãi tiền gửi | Lãi chênh lệch tỷ giá hối đoái đã thực hiện | Lãi từ thanh lý các công ty con | Cổ tức được chia từ Vietjet Air IVB No. I Limited, một công ty con | Cổ tức được chia từ đơn vị khác | Thu nhập tài chính khác
Công ty Cổ phần Hàng không VietJet
Thuyết minh báo cáo tài chính riêng cho năm kết thúc ngày 31 tháng 12 năm 2018 (tiếp theo...
```

**Ví dụ chunk ACB `table_2` (CĐKT ngân hàng):**
```
BÁO CÁO consolidated 2022 | balance_sheet | ACB
 |  | Thuyết minh | 31.12.2022 Triệu VND | 31.12.2021 Triệu VND
TÀI SẢN | Tiền mặt, vàng bạc, đá quý | Tiền gửi tại Ngân hàng Nhà nước | ...
C C C C C C C C C ...        ← ANCHOR NOISE (dòng "C" lặp từ OCR)
NGÂN HÀNG THƯƠNG MẠI CỔ PHẦN Á CHÂU
```

### 5.3. 2 biểu diễn cho mỗi chunk

| | text_dense | text_lex |
|---|---|---|
| Dùng cho | **Dense embedding** (BGE-M3, bge_m3_server local + DeepInfra fallback) | Sparse TF (**đang tắt** — `use_sparse=False`) |
| Xử lý | giữ nguyên, cắt `max_chars=4000` | bỏ dấu + chỉ [a-z0-9], cắt 6000 |

### 5.4. Embedding model (config `local_vllm.yaml`)

| Tham số | Giá trị hiện tại |
|---|---|
| provider chính | `http_bge` (local `scripts/bge_m3_server.py`, BGE-M3, port 8000) |
| fallback | DeepInfra `BAAI/bge-m3` (OpenAI-compatible embeddings API) |
| model | **`BAAI/bge-m3`** (dense 1024-d, fp16) |
| dense_dim | **1024** (bge-m3) |
| max_chars | 2000 |
| batch_size | 100 |
| workers | 12 |

Provider chain **local-first, sticky-first**: thử provider "sticky" (lần trước thành công) trước; lỗi transient (timeout/conn/429/5xx) → nhảy provider kế; cả chain fail → cooldown 30s rồi thử lại. Embed: bge_m3_server local (HTTP `/embed`, không cần key/SDK) → DeepInfra BAAI/bge-m3 (1024-d). **Dim guard**: mọi endpoint embed phải cùng `dense_dim=1024` với primary (Embedder assert lúc init) — tránh embed sai dim làm hỏng Qdrant index.

**Ví dụ vector thật** (query Q1 "Lãi tiền gửi ... VJC ... triệu đồng", dim 1024, chuẩn hoá norm=1):
```
[0.1502, 0.0581, -0.0231, 0.0062, 0.0133, -0.0330, 0.0665, 0.0038, ...]   (1024 giá trị)
```
(Chỉ minh hoạ 8 giá trị đầu — vector thật là mảng 1024 số float32.)

**Lịch sử lựa chọn model:** ban đầu dùng `baai/bge-m3` (OpenRouter trả phí) → thử `nvidia/nemotron-3-embed-1b:free` (dim 2048) một thời gian → **đã revert về bge-m3** (local `bge_m3_server.py` + DeepInfra fallback) vì dim 1024 ổn định, không lệch dim, ít variance hơn. Cấu hình `api.yaml` (cloud-only CŨ: DeepInfra LLM + OpenRouter nemotron embed) giữ lại làm fallback profile, **không còn là default**; `local_vllm.yaml` mới là config chính cho local-AI.

### 5.5. Index Qdrant (Docker `localhost:6333`, collection `bctc_tables`)

- **118,728 points** (bảng n_rows ≥ `min_n_rows=5`, bỏ bảng junk/TOC).
- Dense HNSW (INT8 quantize, cosine), **dim 1024** (bge-m3) + sparse TF (local, `modifier: idf`).
- `use_dense=True`, `use_sparse=False` (dense-only — sparse TF gây nhiễu tiếng Việt).
- **`k=10`** (top-k bảng trả evidence; 5→10 vì bảng đúng hay rank 6-8).
- **Payload** mỗi point: `report_id, ticker, year, report_type, table_id, statement, unit_factor, header_text, row_labels, period_cols`.
- **Search flow**: entity filter (ticker/year/report_type) → hybrid (dense + sparse RRF) → statement bonus mềm → top-k.
- **Fallback** (retrieval): filter report_type quá hẹp (báo cáo `other` không tách cons/sep — EVF/FTS...) → bỏ filter type. *(Lưu ý: đây là fallback tầng retrieval — khác với provider fallback DeepInfra ở tầng serve/embed, xem §5.4.)*

---

## 6. Agent & codegen (`src/vifinqa/agent/loop.py`)

1. **Retrieve**: `pipeline.search(question)` → top-k SearchResult.
2. **Plan evidence**: bảng statement → dùng `evidence_merged` (gộp toàn bộ, preview 20k ký tự); bảng notes → tidy per-table. Dedupe theo (report_id, statement).
3. **Deterministic engine** (`engine/deterministic.py`): lookup đơn giản (1 ticker + 1 chỉ tiêu) → match label trong facts/evidence → sinh `pandas_query` template. Không tốn LLM.
4. **Codegen**: LLM (Qwen3.5-9B qua vLLM local, port 8001, fallback DeepInfra) sinh `pandas_query` từ cards (columns + sample rows + fact hints).
5. **Sandbox**: AST check + exec trong cô lập, retry ≤2 lần khi lỗi (feed error vào LLM).
6. **Builder** (`submission/builder.py`): materialize tidy CSVs flat + rewrite `relevant_tables` → `report_id|<start_line>` + nhúng `vn_num()` self-contained vào query.

---

## 7. Vấn đề đang gặp phải

### Tổng quan 2 lớp lỗi retrieval (đã phân tích từ data thật)

| | Lớp A — Q7 (Quỹ khen thưởng HT1) | Lớp B — Q19 (Tỷ lệ biểu quyết HHV) |
|---|---|---|
| Bảng đúng | `table_7` (CĐKT, có cột kỳ) | `table_1` (notes %, **không cột kỳ**) |
| Trong evidence? | ❌ không vào top-k (row_labels cắt 10 dòng) | ✅ vào evidence (k=10) nhưng **file rỗng 0 rows** |
| Root cause | chunk text thiếu label → dense miss | `grid_to_tidy` period_cols rỗng → không sinh row |
| Quy mô | bảng statement >10 dòng (6,662/10,808) | **25,474 bảng không period_cols** (1,545 bảng %/tỷ lệ) |

### 🔴 Vấn đề #1: `row_labels` chỉ lấy 10 dòng đầu → bảng dài bỏ sót chỉ tiêu

**Hiện trạng:** `catalog_builder.py::header_and_labels` dừng ở `len(labels) >= 10`.
- 66% statement tables (6,662/10,808) có n_rows > 20 → **chỉ tiêu ở dòng 11+ không có trong text search**.
- Ví dụ thực: HT1 2019 "Quỹ khen thưởng, phúc lợi" (mã 322) nằm **dòng 11** của CĐKT → bảng `table_7` **không bao giờ** vào top-k (dù k=20) → câu Q7 trả 0.0.
- Hệ quả: ~5-8% câu hỏi về chỉ tiêu cuối bảng dài bị 0.0.

**Cách sửa triệt để (chưa làm — cần rebuild catalog + index ~1.5h free):**
1. `header_and_labels` → full bảng (hoặc 30 dòng).
2. Đồng bộ giới hạn: `embedding.max_chars` 2000 → 4000-6000, `build_payload` row_labels 600 → 2000.
3. Rebuild catalog (12p) + index (~70p).

**Hướng thay thế rẻ hơn:** lexical fallback index từ `evidence/*.csv` (đã có tidy) — fuzzy match label → union bảng vào evidence bất kể top-k dense. Không rebuild index.

### 🔴 Vấn đề #5: Bảng không có cột kỳ (bảng %, danh sách công ty) → evidence rỗng

**Hiện trạng:** `grid_to_tidy` chỉ sinh row khi có `period_cols`; bảng như HHV `table_1` (Tỷ lệ lợi ích/Biểu quyết/Vốn đầu tư, không cột năm) → **evidence 0 rows**.
- Ví dụ thực: Q19 HHV `table_1` đã nằm trong evidence (k=10) nhưng CSV rỗng → LLM không có dữ liệu → 0.0.
- Quy mô: **25,474 bảng không period_cols**, trong đó **1,545 bảng** có header chứa `%/tỷ lệ/` (dạng sở hữu, biểu quyết, cổ tức).

**Cách sửa (chưa làm):** thêm nhánh "bảng không-kỳ" trong `grid_to_tidy` — chọn cột số chính (≥50% cell parse được, ưu tiên header `%|ty le|so huu|gia tri`), emit row với `ky = str(report_year)`. Chỉ rebuild evidence (~13p), không cần rebuild index.

### 🟡 Vấn đề #2: Deterministic chỉ bắt lookup đơn giản

- `is_complex` (tăng/giảm/tỷ lệ/argmax...) → fallback LLM. Các câu so sánh/tăng trưởng chưa có deterministic path.
- `metric_tokens` cắt từ câu hỏi, nhưng label notes không chuẩn → đôi khi miss dù bảng đúng đã trong evidence (vd Q2 "cho vay khách hàng ngành Thương mại" — label chỉ `thuong mai`).

### 🟡 Vấn đề #3: LLM variance trên model free

- LLM free (trước đây qua OpenRouter, nhiều backend) hay đổi hành vi giữa các lần chạy (temperature 0 nhưng không deterministic tuyệt đối) → cùng câu có lúc đúng lúc 0.0 (vd Q2, Q15 trong các lần test). **Đổi sang vLLM local (greedy, temp 0, 1 backend) giảm đáng kể variance**; DeepInfra chỉ là fallback khi local down.
- Giảm thiểu: deterministic engine bắt trước, retry repair path, `answer_abs_tol`, cache query đã pass sandbox.

### 🟡 Vấn đề #4: Chunk text có noise từ anchor

- `anchor_context` (6 dòng text trước bảng) có thể chứa TOC/trang (`TRANG`, `BÁO CÁO CỦA BAN TỔNG GIÁM ĐỐC`), dòng lặp ký tự (`C C C C...`), tên công ty... → làm nhiễu embedding.
- Ví dụ thật: ACB `table_2` anchor bắt đầu bằng `C C C C C C...` (50+ ký tự "C") rồi mới tới tên ngân hàng.
- Giải pháp (chưa làm): regex filter dòng toàn hoa + không số, dòng khớp `TRANG|MỤC LỤC|KIỂM TOÁN ĐỘC LẬP`. ⚠️ Cần rebuild index vì chunk text đổi.

### 🟢 Đã giải quyết (ghi nhớ tránh lặp lại)

- Wide CSV lệch cột header ngân hàng (rowspan) → parser fix + rebuild toàn corpus.
- `row_labels` lấy nhầm cột STT ("1|2|3") khi code_col=0 → dùng `label_col` từ layout (giảm 4,215 → 12 bảng sai).
- Index mixed embedding (đã thay bằng local `bge_m3_server.py` + DeepInfra fallback, dim 1024 đồng nhất) → rebuild.
- FlagEmbedding 1.4 dependency vỡ (torchvision.io/BloomPreTrainedModel) → pin 1.2.10 + transformers 4.44.2.
- Nemotron embed không hỗ trợ base64 → `encoding_format="float"`. (Đã bỏ nemotron; bge-m3 qua `bge_m3_server.py /embed` trả dense JSON, DeepInfra bge-m3 qua OpenAI embeddings API vẫn dùng `encoding_format="float"`.)
- Qdrant filter AND trả 0 khi chưa có payload index → tạo payload index tường minh (year/ticker/report_type/statement).
- Qdrant "address already in use" khi Docker restart → dùng `free_port` + restart container.

---

## 8. Cách chạy

```bash
# 1. ETL: wide + catalog + layouts (toàn corpus ~12p)
python scripts/run_etl.py --workers 8

# 2. Tidy evidence mọi bảng (~13p)
python scripts/rebuild_evidence.py --workers 8

# 3. Facts → facts_all.csv (~15p)
python scripts/run_facts.py --workers 8

# 4. Statement merged + statement_meta (~2p)
python scripts/run_merged_evidence.py

# 5. Khởi động serving local-AI trên GPU thuê (GPU ≥16GB, lý tưởng 24GB; CHUNG 1 GPU 2 port)
#    1 lệnh spawn bge TRƯỚC (poll /health tới ready) rồi mới start vLLM (tránh tranh VRAM):
python scripts/serve_all.py
#    (hoặc chạy 2 lệnh riêng: bge_m3_server.py rồi vllm_qwen_server.py)
#    In ra block YAML → copy vào configs/local_vllm.yaml.
#    GPU remote → đổi base_url http://localhost thành http://<GPU_IP> trong local_vllm.yaml.
#    GPU yếu/OOM → giảm --gpu-memory-fraction xuống 0.5 hoặc --max-model-len.

# 6. Index Qdrant — BẮT BUỘC rebuild lần đầu (dim 2048 → 1024, drop collection + xoá state):
docker start vifinqa-qdrant   # nếu tắt
python scripts/build_retrieval_index.py --config local_vllm.yaml --rebuild
#    Lần sau (không đổi model/dim) bỏ --rebuild; cache .npy tự invalidate theo model+dim.

# 7. Codegen 20 câu test
python scripts/run_codegen_spots.py --ids 1,2,3,4,6,7,8,9,10,11,12,15,16,18,19,25,27,31,32,36

# 8. Codegen full 1012 → results → submission
python scripts/run_codegen.py
python scripts/build_submission.py
```

Mọi lệnh codegen/pipeline/submit thêm `--config local_vllm.yaml` để dùng local-AI profile.

Yêu cầu: `.env` chứa `DEEPINFRA_TOKEN` (fallback cloud); `VLLM_API_KEY` tuỳ (mặc định `vllm`); `OPENROUTER_API_KEY` chỉ còn dùng cho profile `api.yaml` CŨ. Docker chạy Qdrant (`vifinqa-qdrant`, port 6333).

---

## 9. Cấu trúc repo

```
configs/          base.yaml (chunk/index mặc định) + api.yaml (cloud-only CŨ, fallback profile)
                  + local_vllm.yaml (profile local-AI chính: vLLM + bge-m3 + fallback DeepInfra)
scripts/          ETL, facts, merged, index, codegen, builder
                  + bge_m3_server.py (BGE-M3 embed server, port 8000)
                  + vllm_qwen_server.py (Qwen3.5-9B qua vLLM, port 8001, thinking off)
                  + serve_all.py (spawn bge + vLLM chung 1 GPU, in YAML cho local_vllm.yaml)
src/vifinqa/
  etl/            parser, format_classify, catalog_builder, tidy, facts_builder, merged_evidence
  retrieval/      entity, index (chunk+embed), search (hybrid), pipeline, facts_index
  agent/          loop.py (retrieve→plan→deterministic→codegen→retry)
  engine/         deterministic.py (lookup không cần LLM)
  codegen/        prompt.py, llm.py
  sandbox/        ast_check, executor (chạy pandas_query cô lập)
  submission/     builder, pack, validate
```
