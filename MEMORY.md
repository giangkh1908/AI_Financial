# MEMORY.md — GURU · ViFinQA (Knowledge Base of Claude)

## 0. Bản chất tài liệu này

Đây là tài liệu ghi nhớ toàn diện của Claude cho dự án GURU · ViFinQA, **bổ sung** cho `CLAUDE.md` (tài liệu dài hạn của dự án). Trong khi `CLAUDE.md` ghi quy định cuộc thi/dữ liệu/phương pháp/kế hoạch, bản MEMORY này tập trung vào: **map mã nguồn chi tiết** (file/function/getcha), **kiến trúc & luồng pipeline**, **schema dữ liệu & outputs hiện có trên đĩa**, **cấu hình & knobs**, **trạng thái milestone**, **phiên làm việc**. Một phiên Claude mới đọc file này + `CLAUDE.md` sẽ nắm ngay bối cảnh, vị trí code, trạng thái hiện tại, việc cần làm kế tiếp mà không phải đọc lại toàn bộ codebase.

**📌 TÀI LIỆU THAM KHẢO CHÍNH (cập nhật 9/8/2026):**
- `README.md` — **định dạng dữ liệu 7 tầng** (mỗi tầng có ví dụ thật từ corpus), schema, chunking/embed, vấn đề đang gặp.
- `ARCHITECTURE.md` — **kiến trúc & luồng** (ETL/Retrieval/Answering), module map, config, quyết định thiết kế.
- Đọc 2 file này TRƯỚC khi đọc phần còn lại của MEMORY — chúng là nguồn sự thật mô tả hệ thống hiện tại.

> ⚠️ **Lưu ý version:** Các mục 2-8 phía dưới ghi lại code/lịch sử TỪ TRƯỚC đợt refactor 8/8-9/8 (một số chi tiết đã lỗi thời: bge-m3 1024-d, catalog 16 cột, rowspan chưa xử lý, `use_sparse=true`, evidence 7,469 file...). **Trạng thái HIỆN TẠI là §0.1** — khi xung đột, tin §0.1 + README/ARCHITECTURE. Các mục cũ giữ lại vì chứa bài học/incident/lịch sử nộp bài hữu ích.

## 0.1. HIỆN TRẠNG KIẾN TRÚC (cập nhật 9/8/2026) — đọc trước tiên

> ⚠️ Các phần dưới (mục 2-8) mô tả code/lịch sử TRƯỚC đợt refactor 8/8-9/8. Phần này là trạng thái HIỆN TẠI, ưu tiên cao nhất.

### 0.1.1. Kiến trúc cốt lõi: canonical wide làm nguồn sự thật

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

**Nguyên tắc sống còn:** mọi consumer (facts/tidy/merged/index/codegen) đọc `TableLayout` từ catalog/layouts — **cấm tự đoán cột lại**. Bug lịch sử (lệch cột bank header) do mỗi nơi re-heuristic → đã fix bằng 1 nguồn sự thật.

### 0.1.2. Dữ liệu derived hiện tại (9/8)

| Đường dẫn | Số lượng | Ghi chú |
|---|---|---|
| `tables/{rid}/table_{N}.csv` | 146,246 | CANONICAL WIDE, rowspan-aligned, entity-decoded |
| `catalog_tables.csv` | 146,246 dòng | **23 cột** (thêm `header_idx, period_row_idx, code_col, label_col, period_cols, thuyet_minh_cols, number_format`) |
| `layouts/{rid}.json` | 1,973 | TableLayout chi tiết mọi bảng (kể cả notes) |
| `evidence/{rid}__table_N.csv` | 146,246 | Tidy mọi bảng `[chi_tieu, Mãsố, ky, value]` (đã rebuild TOÀN BỘ) |
| `evidence_merged/` | 5,474 | Statement gộp |
| `facts_all.csv` | 400,510 dòng | 12 cột (item_code/label/value_vnd/src_table_ids) |
| `statement_meta.csv` | 5,474 | report_id/statement/src_table_ids |
| `embeddings/` | 200 | `.npy`+`.json` per ticker, **dim 2048** |
| Qdrant `bctc_tables` | **118,728 points** | **dense 2048-dim**, INT8 quantize, `use_sparse=False` |

### 0.1.3. Chunking & Embedding (9/8)

- **1 bảng = 1 chunk** (118,728 bảng = 118,728 points).
- Chunk text = `prefix + header_text + row_labels + anchor_context + fact_labels` (nối `\n`, cắt `max_chars=2000`).
- **`row_labels` chỉ 10 dòng đầu** cột label_col ⚠️ → Vấn đề #1 (bảng dài bỏ sót chỉ tiêu cuối).
- **Model: `nvidia/nemotron-3-embed-1b:free`** (OpenRouter, free, **dim 2048**) — thay bge-m3 (1024) sau verify tốt hơn tiếng Việt. Lịch sử: bge-m3 → GPU thuê RTX 3060 (OOM) → nemotron free. (→ đã thay: quay lại bge-m3 1024-d local + DeepInfra fallback, xem §0.1.7)
- **`use_dense=True, use_sparse=False`** (dense-only — sparse TF gây nhiễu tiếng Việt).
- **`k=10`** (5→10 vì bảng đúng hay rank 6-8).
- Cache embed: `embeddings/{ticker}.npy` + hash md5 text chunks → resume không embed lại.

### 0.1.4. Luồng answer (2 tầng)

1. **Deterministic engine trước** (`engine/deterministic.py`) — lookup đơn giản (1 ticker + 1 chỉ tiêu), match label trong facts/evidence → sinh pandas_query template, **không tốn LLM**. Bắt ~1/3 câu.
2. **Codegen LLM** (`codegen/`) cho câu phức (so sánh/tỷ lệ/argmax/notes không chuẩn) → sinh `pandas_query` → **Sandbox** (AST check + exec cô lập `python -I`) → retry ≤2 lần.
3. **Builder** (`submission/builder.py`) — materialize tidy CSVs flat + rewrite `relevant_tables` → `report_id|<start_line>` + nhúng `vn_num()` self-contained.

### 0.1.5. Verify kết quả (9/8, 30 câu deterministic, no-LLM)

**14/30 deterministic match** với nemotron (so bge-m3 trước đó 5/30). Các câu then chốt đúng: Q1 (lãi tiền gửi VJC 208,253), Q8 (lương FTS 30.69), Q15 (thù lao MPC 150), Q36 (lợi thế TM VRE 624,529).

### 0.1.6. 2 lớp lỗi retrieval (phân tích từ data thật 9/8)

| | Lớp A — Q7 (Quỹ khen thưởng HT1) | Lớp B — Q19 (Tỷ lệ biểu quyết HHV) |
|---|---|---|
| Bảng đúng | `table_7` (CĐKT, có cột kỳ) | `table_1` (notes %, **không cột kỳ**) |
| Trong evidence? | ❌ không vào top-k (row_labels cắt 10 dòng) | ✅ vào evidence nhưng **file rỗng 0 rows** |
| Root cause | chunk text thiếu label → dense miss | `grid_to_tidy` period_cols rỗng → không sinh row |
| Quy mô | bảng statement >10 dòng (6,662/10,808) | **25,474 bảng không period_cols** (1,545 bảng %/tỷ lệ) |

**Hướng fix ưu tiên:** (1) tidy bảng không-kỳ (fix Q19-class, rebuild evidence ~13p); (2) lexical fallback index từ evidence (fix Q7-class, không rebuild index); (3) anchor noise filter (rebuild index ~70p); (4) deterministic mở rộng fuzzy (difflib stdlib, không thêm rapidfuzz).

### 0.1.7. Local-AI + fallback (15/8/2026) — CURRENT

> ⚠️ Phần này là trạng thái MỚI NHẤT về serving/embed/LLM. Các mục cũ bên dưới (§0.1.3, §5, §6…) vẫn dùng nemotron-2048/OpenRouter — đã bị phần này thay.

**Serving local (GPU thuê ≥16GB, lý tưởng 24GB, CHUNG 1 GPU 2 port):**
- **Embed:** `scripts/bge_m3_server.py` — BGE-M3 (`BAAI/bge-m3`), dense **1024-d**, fp16, ~10-20% VRAM, **port 8000**. API: `POST /embed {texts,return_dense}` → `{dense: [[...]]}`, `POST /embed_bin` (bytes nhanh 5x), `GET /health`.
- **LLM:** `scripts/vllm_qwen_server.py` — `Qwen/Qwen3.5-9B` qua vLLM (OpenAI-compatible), `--gpu-memory-fraction 0.65` (60-70% VRAM), **port 8001**. Thinking (Qwen3 reasoning) **TẮT**: server không thêm `--enable-reasoning`; client gửi `chat_template_kwargs.enable_thinking=False` qua `extra_body`. API: `POST /v1/chat/completions`, `GET /health`.
- **1 lệnh khởi động cả 2:** `python scripts/serve_all.py` — spawn bge **TRƯỚC**, poll `GET /health` tới ready (≈model load xong) rồi mới start vLLM (tránh tranh VRAM lúc load), in block YAML cho `configs/local_vllm.yaml`, Ctrl-C tắt sạch cả 2 process group (vLLM grandchild cũng chết, không leak GPU). Cờ: `--host --embed-port --llm-port --gpu-memory-fraction --llm-model --max-model-len --embed-batch-size --embed-max-length --embed-timeout --llm-timeout --no-llm --no-embed`. Còn ~15% VRAM headroom; GPU yếu/OOM → giảm `--gpu-memory-fraction` xuống 0.5 hoặc `--max-model-len`. Có thể vẫn chạy 2 lệnh riêng: `bge_m3_server.py` rồi `vllm_qwen_server.py` (`serve_all.py` chỉ gói 2 cái đó).

**Fallback chain (local-first, sticky-first):**
- Quy tắc: thử provider "sticky" (lần trước thành công) trước; lỗi transient (timeout/conn/429/5xx) → nhảy provider kế; cả chain fail → cooldown 30s rồi thử lại. Retry-in-provider nhỏ (1-2) rồi mới nhảy.
- **LLM:** vLLM local (Qwen3.5-9B) → DeepInfra `Qwen/Qwen3.5-9B` (thinking off qua `extra_body {reasoning:{enabled:false}}`).
- **Embed:** `bge_m3_server` local (provider `ngrok`/`http_bge`, HTTP `/embed`, KHÔNG cần key/SDK) → DeepInfra `BAAI/bge-m3` (OpenAI-compatible embeddings API, 1024-d).
- **Dim guard:** mọi endpoint embed phải cùng `dense_dim=1024` với primary (Embedder assert lúc init) — tránh embed sai dim làm hỏng Qdrant index.

**Config:** `configs/local_vllm.yaml` (deep-merge `configs/base.yaml`) — chỉ override `llm`/`codegen_llm`/`retrieval.embedding` (provider + fallbacks); kế thừa tuning base (k=10, max_chars=4000, use_sparse=false, rerank.enabled=false). DeepInfra `api_key` rỗng → tự resolve env `DEEPINFRA_TOKEN`. vLLM `api_key` default `vllm` (hoặc env `VLLM_API_KEY`). `configs/api.yaml` (profile cloud-only CŨ: deepinfra LLM + openrouter nemotron embed, không fallback) — GIỮ lại làm fallback profile, KHÔNG còn là default; `local_vllm.yaml` mới là config chính cho local-AI.

**Config schema (`src/vifinqa/config.py`):** thêm `ProviderRef` (LLM fallback) + `EmbedProviderRef` (embed fallback), `extra='forbid'`, `_env_api_key` resolve theo provider; `LLMConfig.fallbacks` + `EmbeddingConfig.fallbacks`.

**Đổi `dense_dim` 2048 → 1024** (nemotron cũ → bge-m3) ⇒ BẮT BUỘC rebuild Qdrant collection + re-embed ~118K điểm: `python scripts/build_retrieval_index.py --config local_vllm.yaml --rebuild` (drop collection + xoá `retrieval_state.json`; cache `.npy` tự invalidate theo model+dim nên không cần xoá tay).

**Code đã đổi:** `config.py` (`ProviderRef`/`EmbedProviderRef`/`fallbacks`/`_env_api_key`), `codegen/llm.py` (`LLMClient` multi-provider chain `_order`/`_try_provider`/`_call_chain`), `retrieval/index.py` (`Embedder` + `make_embedder`, thay `_new_embed_client` vốn hardcode OpenRouter), `retrieval/pipeline.py` + `scripts/build_retrieval_index.py` (gọi `make_embedder`, `--rebuild`), `scripts/vllm_qwen_server.py` (MỚI), `scripts/serve_all.py` (MỚI), `configs/local_vllm.yaml` (MỚI).

**LlamaIndex:** ĐÃ BỎ — chỉ ~20% pipeline thay được (commodity layer); `QdrantVectorStore` mất native RRF; không có primitive cho entity-extract/deterministic/label-recall; sandbox grader-parity hiện tại hơn mọi lựa chọn code-exec.

Quyết định trước đây chốt nemotron-2048/OpenRouter (§0.1.3/§0.1.5/§5/§6 bên dưới) ĐÃ BỊ THAY bởi local-AI+DeepInfra-fallback này.

## 1. Tổng quan cuộc thi & mục tiêu

**Bài toán:** Financial Table Retrieval & Text-to-Pandas Query Generation trên BCTC doanh nghiệp niêm yết Việt Nam.

**Hai nhiệm vụ cốt lõi:**
1. **Table Retrieval:** Với câu hỏi `q`, xác định tập con bảng `D' ⊂ D` trong kho BCTC mà mỗi bảng chứa số liệu cần tính đáp án.
2. **Text-to-Pandas:** Dựa trên bảng đã truy hồi, sinh pandas query chạy được, trả đúng số liệu.

**5 yêu cầu BTC:** (1) Truy hồi chính xác — đúng công ty/năm/bảng/vị trí; (2) Hiểu truy vấn tài chính tiếng Việt (so sánh đa công ty/năm/chỉ số dẫn xuất ROE/ROA/tăng trưởng); (3) Sinh pandas query đúng logic/schema/đơn vị/kỳ; (4) Dẫn nguồn minh bạch (công ty/năm/tên báo cáo/tên bảng/vị trí); (5) Kiểm soát hallucination.

**Quy mô dữ liệu:** 1,012 câu hỏi, 1,973 báo cáo OCR từ 100 công ty niêm yết, giai đoạn 2015–2025. Kho bảng ~146,246 wide tables (ETL M1), trong đó 10,797 bảng BCTC statement (BS 5,066 / IN 2,349 / CF 3,382).

**Định dạng nộp bài (OFFICIAL SPEC — authoritative 6/8/2026):**
```
submission.zip
├── submission.json   (mảng 1,012 record)
└── data/              (CSV evidence flat, csv_path bắt đầu "data/")
```
Record: `id (int), question (str), answer (float), relevant_docs [<report_id>], relevant_tables ["<report_id>|<line>"], evidence [{variable, csv_path}], pandas_query (str)`.
- `relevant_docs`: `report_id` = tên file cuối trong path bỏ `.txt`. VD path `ocr_filter\AAA\2015\AAA_financial_statements_2015_consolidated` → `AAA_financial_statements_2015_consolidated`.
- **`relevant_tables` = `"<report_id>|<vị trí bảng trong báo cáo>"`**, vị trí = **số dòng bắt đầu của bảng trong file OCR** (physical line number). VD spec: `AAA_financial_statements_2015_consolidated|350`. **Đây là FORMAT DUY NHẤT đúng per OFFICIAL SPEC**, KHÔNG phải `report_id|table_N`.
- `evidence.variable`: Python identifier hợp lệ, unique trong câu hỏi. `csv_path`: relative bắt đầu `data/`, file phải tồn tại trong zip. **Filename là tuỳ chọn** (spec ví dụ `data/AAA_financial_statements_2015_consolidated_table_1.csv`) → ta dùng `data/{rid}__table_N.csv` (vẫn dùng `table_N` trong filename, chỉ `relevant_tables` VALUE là line format).
- `pandas_query`: chạy lại được trên CSV đã nộp. ⚠️ Spec ví dụ dùng **bare `df1`** (`df1[(df1.company=='VNM') & (df1.year==2023)]['net_revenue'].values[0]`) + variable "được sử dụng trực tiếp trong pandas_query" → mâu thuẫn contract `dfs["<table_ref>"]` của Fix B — xem §8 "Submission format gotchas" (chưa resolve, cần shim dual-compat trước khi rebuild). Ví dụ spec cũng self-inconsistent (hỏi VNM nhưng doc/CSV AAA) — chỉ tin FORMAT, không tin giá trị.

**Đánh giá (macro-average):** Retrieval Precision/Recall, **F2 = (5·P·R)/(4·P+R)** (ưu tiên Recall), Answer Accuracy (tol tuyệt đối 0.01 theo paper; ngưỡng chính thức do BTC), Execution Accuracy (code chạy + đúng / tổng). Thiếu file/thiếu câu → bài không đánh giá, không tính quota.

## 2. Map mã nguồn (code map)

### `src/vifinqa/etl/` — ETL chuẩn hoá OCR → bảng
- `__init__.py` — package marker, docstring only.
- `numbers.py` — parse số/đơn vị/label/period từ cell OCR. `UNIT_FACTORS` (nghìn/triệu/tỷ/trăm/đồng), `_HEADER_UNIT_RE`, `_UNIT_LINE_RE`, `_YEAR_DATE_RE`, `_PERIOD_LABEL_RE` (excludes "Số năm"), `_NUMBER_RE` (`(x)`/`-x` âm). Hàm: `parse_vn_number`, `parse_number(s, thousands, decimal)`, `detect_number_format` (en `,`nghìn vs vi `.`nghìn), `normalize_label` (NFD + manual Đ/đ), `unit_factor_from_label`, `detect_unit` (anchor-bounded không whole-page), `parse_period_header` (year_end/year_start/flow_year/restated_cur), `is_period_cell`.
- `parser.py` — split trang/extract HTML table/build grid. `PAGE_RE`, `TABLE_RE`. `split_pages`, `extract_tables`, `parse_table_grid` (BeautifulSoup lxml, **colspan expand** text duplicate + **rowspan handled** — fix session 8/8: header ngân hàng `rowspan=2 colspan=2` lệch cột → expand rowspan qua `active: dict[col→(text, remaining)]`), `find_header_row`. Dataclass `Page`, `TableGrid`.
- `statements.py` — classify 3 statement vs notes + M2 fragment merge/fact emit. `_STMT_TITLE_RE` (VN+EN, bank variant `bao cao tinh hinh tai chinh`), `_NEGATIVE_RE`, `_VAS_CODE_RE` (`^\d{1,3}[a-z]?$`), `_BANK_STT_RE` (Roman/letter/1-2 digit, **accept lowercase** a/b/c/g), `_FORMULA_RE`, `_CARRY_RE` (defined, unused). Hàm: `_has_statement_structure` (REQUIRED period col + header Mã số/STT+Chi tiêu OR **≥50% code col** + ≥3 data rows), `classify_statement`, `find_item_code_col` (≥70% match, densest col), `_period_columns` (gồm `period_label`), `_label_column` (non-period/non-code, highest avg text length), `header_signature`, `group_statement_fragments` (gom consecutive cùng stmt+signature), `build_asset`, `emit_facts` (dedupe item_code ở biên fragment, label inheritance, `value_vnd = raw × unit_factor`), `validate_asset` (log formula-label, cross-sum BS 270==440 tol 1.0, giữ giá trị gốc). Dataclass `Fragment`, `StatementAsset`.
- `catalog_builder.py` — every table → wide raw CSV + catalog row + documents. `ANCHOR_LINES=6`. Hàm: `anchor_text`, `write_table_csv` (UTF-8, rows padded, **raw OCR strings preserved**), `header_and_labels` (skip section-title rows `len(set(row))==1`, **`label_col` từ layout — fix session 8/8: trước lấy nhầm cột STT "1|2|3" khi code_col=0**, ⚠️ cắt **10 dòng đầu** → Vấn đề #1), `process_report`, `write_catalog_csv`, `write_documents_csv`, `build_catalog`, `merge_catalog_parts`, `write_layouts_json` (**MỚI 8/8 — ghi `layouts/{rid}.json`**), `layout_to_catalog_fields`, `_period_cols_str`, `_json_list`. **`CATALOG_HEADER` 23 cols** — sau 16 cols cũ + 7 cột layout (`header_idx, period_row_idx, code_col, label_col, period_cols, thuyet_minh_cols, number_format`). **`DOC_HEADER` 7 cols**. Dataclass `CatalogRow` có field `start_line: int` + `layout_dict()` method. `table_start_lines(full_text)` trả `{table_idx(1-based): physical line của <table>}` (dùng cho `relevant_tables` submission line format). `process_report` cũng detect layout cho **notes/segment** (vd ACB table_37 cho vay theo ngành: period_cols=[1,2]) — không chỉ statement.
- `facts_builder.py` — Tier-A facts 3 BCTC → long-format CSV VND. **Self-contained** (re-classify/detect/parse cùng M1 cho nhất quán). **`FACTS_HEADER` 11 cols** (`ticker, year, report_type, statement, item_code, item_label, item_label_raw, period_key, period_label, value_vnd, src_table_ids`). Hàm: `_scan_report_tables` (table_idx 1-based across whole report), `build_report_facts` (scan→group→build_asset+emit_facts), `write_facts_csv`, `write_facts_part`, `merge_facts_parts`. `src_table_ids` = single fragment table_id per fact.
- `tidy.py` — wide → tidy `[chi_tieu, Mãsố, ky, value]` cho grader-stable query. `_CODE_COLS`, `_SKIP_COLS`, `_OPENING_HINTS` (drop opening balance cols), `_YEAR_RE`. Hàm: `report_year`, `_period_year`, `wide_to_tidy` (legacy — read `index_col=0`), **`grid_to_tidy` (MỚI 8/8 — raw grid header=None + layout dict: label_col/code_col/period_cols/unit_factor/period_row_idx)**, `load_layout_dict` (đọc `layouts/{rid}.json`), `wide_csv_to_tidy` (dùng grid_to_tidy + layout), `write_tidy_csv` (`float_format="%.6f"`). ⚠️ **Bảng không period_cols → rỗng 0 rows** (Vấn đề #5 — fix pending). **Parse VN only** — English BCTC cần unit_factor pre-scaled.
- `format_classify.py` (**MỚI 8/8 — NGUỒN SỰ THẬT layout**): `@dataclass TableLayout(header_idx, period_row_idx, code_col, label_col, period_cols, thuyet_minh_cols, unit_factor, unit_label, number_format)`, `detect_layout(grid, anchor_text, report_type)` (header marker "Mã số"/"Chỉ tiêu"/"STT"/"Codes", period = cột có năm; header 2 tầng ngân hàng → `period_row_idx=header_idx+1`; lọc cột "Tập đoàn"/"Công ty" theo report_type cho MSR 2015-2018), `classify_table` (statement qua `statements.classify_statement` + layout). Notes/segment cũng detect layout (period cols + unit).

### `src/vifinqa/retrieval/` — Retrieval hybrid
- `entity.py` — extract entity từ câu hỏi VN. `@dataclass Entities` (tickers/years/year_ranges/report_type/statement/unit_factor/unit_label/matched_names), `CompanyMap`. Hàm: `load_company_map` (name variants longest-first, alias filter corpus, bare_re regex), `extract_tickers` (4-stage: company-name→alias→bare→parenthesized, spans suppress bare), `extract_years` (range expand, joint clamp [2015,2025]), `extract_report_type` (`hợp nhất`→consolidated, `công ty mẹ`→separate), `extract_statement_hint` (soft, income→cash_flow→balance_sheet), `extract_units` (nghìn tỷ→tỷ→...→đồng), `extract_entities`. `COMPANY_ALIASES` (bidv→BID, vietcombank→VCB, hoa phat→HPG, techcombank→TCB dropped at load vì ∉corpus). Statement = **soft hint only**, không hard filter.
- `index.py` — offline build: chunk + dense embed (bge-m3 OpenRouter, cache `.npy` per ticker) + sparse TF + Qdrant upsert. Hàm: `point_id` (uuid5 idempotent), `make_qdrant_client` (local/server), `parse_position`, `tokenize`, `_term_id` (MD5 2^21 bucket stable), `tf_sparse`, `build_table_chunks` (prefix+header+labels+anchor+fact_labels, truncate max_chars), `build_payload`, `iter_catalog_tables` (filter ticker/embed_statement_only/min_n_rows), `load_fact_labels`, `embed_dense` (batch, ThreadPoolExecutor workers, **5 attempts (4 retries)** exp backoff+jitter trên 429/5xx/timeout), `ensure_collection` (dense COSINE HNSW + optional INT8 quantization + sparse IDF), `build_ticker` (cache key hash(texts)+n+model+dim, upsert batch 256). `_UUID_NS`, `_SPARSE_BITS=21`.
- `search.py` — hybrid search Qdrant native RRF + entity filter + statement bonus. `@dataclass SearchResult` (frozen, `score`, `dense_score`, `sparse_score`, `rank`), `relevant_tables_key()` → `f"{report_id}|table_{position}"` (key nội bộ cho grader dfs dict; **KHÔNG phải format nộp** — builder rewrite sang line format ở packaging). Hàm: `build_payload_filter` (must: ticker MatchAny, year MatchAny, report_type MatchValue; None=global), `hybrid_search` (prefetch dense+sparse limit rerank_depth, `FusionQuery(Fusion.RRF)`), `apply_statement_bonus` (soft add, notes không penalty, re-sort), `_replace_score` (via `dataclasses.replace`).
- `facts_index.py` — đọc facts_all.csv 377K rows. `parse_report_id` regex `^([A-Z0-9]+)_financial_statements_(\d{4})_(consolidated|separate|aggregated|other)` → dạng **`TICKER_financial_statements_YEAR_type[_suffix]`**. `class FactsIndex` (`get_facts`, `facts_for_table` exact match `src_table_ids == table_id` không substring, `table_fact_coverage`, `verify`). **`facts_all.csv` không có cột `report_id`** — không phân biệt HDB `_separate` vs `_separate_1` (M3 deferred).
- `pipeline.py` — orchestrate. `class RetrievalPipeline` (`__init__` load CompanyMap+QdrantClient+OpenAI embed, lazy reranker; `search` extract→filter→embed→hybrid_search→statement_bonus→truncate rerank.candidates→rerank nếu enabled→top k; `_rerank` pairs score sort; `close`). Rerank lazy-import để serve không cần torch.
- `rerank.py` — `LocalReranker` Qwen3-Reranker-0.6B CPU. **Disabled** (M3). `AutoModelForCausalLM` + "yes"/"no" score tokens (không SequenceClassification). `_DTYPE_KWARG={"dtype": torch.float32}` defined nhưng **không apply** trong `from_pretrained` (bug, không impact vì rerank off).

### `src/vifinqa/codegen/` — Text-to-Pandas LLM
- `prompt.py` — build chat messages (system+few-shot+user) enforce **grader contract**: namespace `pd`+`vn_num`+safe builtins (no np/math/re/json), 1 table→`df`, N tables→`dfs["{table_ref}"]` verbatim key (table_ref string thật, VD `dfs["HPG_..._consolidated|table_6"]`, không bare df1/df2), CSV cells strings dtype=str index numeric RangeIndex, schema 4 cols tidy `chi_tieu/Mãsố/ky/value`, end `result = round(<float>, 2)`. `_SYSTEM`, `_FEW_SHOT` (2 ví dụ: single `df` + multi `dfs["HPG_...|table_6"]`), `_format_table_card`, `build_messages`.
- `llm.py` — `LLMClient` OpenRouter `qwen/qwen3.5-9b`. `generate_query(messages, max_retries=2)`. `extra_body={"reasoning":{"enabled":False}}` (`thinking=False` — Qwen3 reasoning default ăn max_tokens 4258 tok → empty content). `_extract_code` pipeline: strip think → last ```python fence → strip imports → strip `def vn_num` (builder inject `_VN_NUM_DEF` cho grader) → strip `dfN = <synthetic>` reassign (giữ `df1 = df1[...]` transforms). OpenAI client `max_retries=1` (hardcoded — fix B). Retry loop `range(max_retries)` = 2 total attempts (1 initial + 1 retry) trên `APITimeoutError`/`APIConnectionError` sleep `2.0+random(0,1)`, `APIStatusError` nếu status ∈ {429,500,502,503,504}.

### `src/vifinqa/sandbox/` — thực thi cô lập
- `ast_check.py` — AST walk chặn `open/eval/exec/read_csv/__import__/__class__`/module `os,sys,subprocess,...`; `check_code(code, max_code_len=4000, max_ast_nodes=300)` (đây là **function-parameter defaults**; runtime callers `agent.loop:177` và `build_submission→validate:63` pass `cfg.sandbox` values **8000/800**, nên **effective runtime limits = 8000/800**, không phải 4000/300). `_BLOCKED_CALLS`, `_BLOCKED_MODULES`, `_BLOCKED_ATTRS`. Cho phép `pd/np/math/re/json` attr nhưng runner chỉ inject `pd`+`vn_num` → np/math/re/json NameError (mirror grader). Docstring không nhắc 6000/500.
- `runner.py` — subprocess `python -I` (isolated, drops PYTHONPATH/user site). **Mirror grader BTC exact**: `pd.read_csv(dtype=str, keep_default_na=False, index_col=None, encoding=utf-8-sig)`, DataFrame→dict `dfs` keyed theo `table_ref="{report_id}|table_N"`, alias `df = dfs[key]` khi 1 CSV. Globals `g = {__builtins__: _SAFE_BUILTINS, pd, vn_num, dfs, print}` (inject `g["print"] = _safe_print`). `vn_num` inline copy `parse_vn_number` (runner không import vifinqa dưới -I). `_SAFE_BUILTINS` trừ `_UNSAFE_BUILTINS` (open/eval/exec/compile/getattr/...). `main` đọc stdin JSON `{code, evidence{table_ref:abs_path}}`, exec, `_to_float(result)` (scalar/Series 1-cell/bool/int/float/numeric string strip `,`), output JSON `{ok, result, error, stdout}`.
- `executor.py` — `run_pandas(code, evidence, root, timeout=20)` spawn runner. `evidence={table_ref: csv_path}` (keys là table_ref string, match runner's dfs dict — không phải variable name). `subprocess.run([sys.executable, "-I", runner], input=payload, capture_output, text, encoding=utf-8, errors=replace, timeout)`. TimeoutExpired → `{ok:False, error:"timeout sau 20s"}`.

### `src/vifinqa/agent/` — loop giải 1 câu
- `loop.py` — `solve(question, qid, pipeline, facts_index, llm, cfg, max_retries=1)`: (1) retrieve `pipeline.search`; (2) build table cards qua `_build_table_card` (gọi `_ensure_tidy` wide→tidy cached `derived/evidence/{rid}__{tid}.csv`, preview 8 rows, fact_hints 25); (3) codegen `build_messages`+`llm.generate_query`; (4) retry loop `attempt in range(max_retries+1)`: `_bad_refs` (cấm bare `df\d+` → NameError grader, `dfs["X"]` key phải thuộc table_refs) → repair; `check_code` AST (pass `cfg.sandbox.max_code_len/max_ast_nodes`=8000/800) → repair; `run_pandas` → answer; (5) fallback `result=0.0` nếu fail hết; (6) `_make_record` schema §3.1 (`relevant_docs` dedupe, `relevant_tables` `r.relevant_tables_key()` = `report_id|table_N` ở record nội bộ, evidence `variable=df{i+1}`, `csv_path=data/{rid}__{tid}.csv`, field nội bộ `_ok/_error`). `_repair` feed error vào LLM + reminder contract.

### `src/vifinqa/submission/` — đóng gói
- `builder.py` — `results.jsonl → submission.json + data/` (tidy CSV 4 cols, không wide). Assert đủ 1012 (thiếu→fallback 0.0). `_VN_NUM_DEF` (def vn_num không `re.compile` — ast_check chặn compile) nhúng vào đầu mỗi query non-trivial qua `_with_vn_num` (bỏ qua fallback `result=0.0`) vì grader BTC không inject helper. `_load_unit_factors` đọc `catalog_tables.csv`. Tidy thiếu → **regenerate từ wide** `wide_csv_to_tidy(wide, uf)` + `write_tidy_csv` (rỗng→header-only, query→0.0). Bỏ `_ok/_error`. **Mới (session 6/8 — fix format `relevant_tables`):** `_load_start_lines(derived_dir) -> {(report_id, table_id): start_line}` đọc cột `start_line` từ `catalog_tables.csv`; `_table_ref_to_line(key, start_lines)` convert `"rid|table_N"` → `"rid|<start_line>"` (fallback original nếu start_line missing/0). `build()` load start_lines và rewrite mỗi record `relevant_tables` sang line format (builder.py:219-223), đếm `relevant_tables_rewritten`. `csv_path` **không đổi** (vẫn `data/{rid}__{tid}.csv`, filename giữ `table_N`). `build()` trả `{n, materialized, tidy_regen, relevant_tables_rewritten, missing_ids}`.
- `validate.py` — re-exec mỗi query trên CSV đã đóng gói, so `answer` tol `abs_tol=0.01`. `_table_ref_from_csv_path` `data/{rid}__table_{N}.csv` → `{rid}|table_N` (mirror grader). Short-circuit `result = 0.0` không spawn. ThreadPoolExecutor workers=8. `validate()` defaults `max_code_len=6000, max_ast_nodes=500` (khác ast_check's own defaults 4000/300; cũng bị `build_submission` override bằng cfg 8000/800). `validate()` trả `{total, ok, crash, mismatch, no_evidence, bad_path}`. `check_code` AST trước, `run_pandas` sau.
- `pack.py` — `pack(out_dir)` → `submission.zip` (json + data root, đúng 1 .json enforced, sort deterministic, ZIP_DEFLATED) + `checksum.txt` audit.

### `src/vifinqa/` core
- `config.py` — `ROOT=D:\GURU`, `_load_dotenv`. Pydantic models `extra="forbid"`: `LLMConfig` (provider/base_url/api_key/model_id=qwen/qwen3.5-9b/temperature=0.0/max_tokens=4096/timeout/thinking=False/**retries=3 default**/extra_headers), `EmbeddingConfig` (baai/bge-m3/1024-d), `SparseConfig`, `QdrantConfig` (mode=server/host=localhost/port=6333/HNSW/quantize), `RerankConfig` (enabled=True default nhưng base.yaml override false), `RetrievalConfig` (k=10/rerank_depth=100/statement_bonus=0.05 default/embed_statement_only=False default/min_n_rows=5), `SandboxConfig` (timeout=20/max_code_len=8000/max_ast_nodes=800), `Config` (paths/retrieval/sandbox/answer_abs_tol=0.01/step_budget=10/llm). `Config.load` deep-merge base+override (api.yaml over base.yaml) via `_deep_merge` (recursive, `extra.model_dump(exclude_unset=True)`).
- `loader.py` — `load_stocks` (code_stock.csv utf-8-sig), `load_questions` (jsonl 1:1 order), `infer_report_type`, `iter_reports` (walk financial_statements/{TICKER}/{YEAR}/{report_id}/{report_id}_extracted.txt → sorted ReportMeta).
- `constants.py` — `UNIT_FACTORS`, `ANSWER_ABS_TOL=0.01`, `STEP_BUDGET=10`, `SANDBOX_TIMEOUT=20`, `MAX_CODE_LEN=4000`, `MAX_AST_NODES=300`, `K=10`, `RERANK_DEPTH=100`. (Constants `MAX_CODE_LEN=4000`/`MAX_AST_NODES=300` diverge với base.yaml 8000/800 — config wins runtime.)

### `scripts/`
- `run_etl.py` — M1 ETL wide+catalog+documents. Checkpoint per ticker `etl_state.json`, catalog parts per ticker merge, ThreadPoolExecutor workers=6. CLI `--workers --config --tickers`.
- `run_facts.py` — M2 facts tier 3 BCTC. Checkpoint per ticker `facts_state.json`, parts merge `facts_all.csv`, ThreadPoolExecutor. CLI `--workers --config --tickers --no-merge`.
- `run_retrieval.py` — run retrieval N câu (default 10) → `retrieval_topk.csv` + `retrieval_metrics.json`. CLI `--limit --spot-ids --out --no-rerank --config` (default `api.yaml`).
- `run_codegen.py` — batch codegen 1012 → `data/out/results.jsonl` (append, checkpoint per id). CLI `--limit --workers --config --out`. **`WALL_CAP = cfg.sandbox.timeout * 6.0 = 120s`** per-question wall-clock guard via daemon thread + `queue.get(timeout=WALL_CAP)`. Timeout → fallback record `answer=0.0`, worker freed. `_load_done` skip done. `lock_file` defined unimplemented.
- `build_submission.py` — 3-stage: build → validate → pack. CLI `--config --results --out --tol --no-strict`.
- `backfill_start_line.py` — **MỚI session 6/8**: đọc OCR của 1,973 report, compute `table_idx→line`, join vào `catalog_tables.csv` thêm cột `start_line`. **Không** re-write 146K wide CSV. Idempotent. Đã chạy: 146,246/146,246 rows có `start_line != 0`, 0 report missing OCR. catalog_tables.csv nay 16 cols.
- `smoke_fmt_100.py` — **MỚI session 6/8 (chưa chạy — user interrupt)**: build submission subset 100 câu (prefer `answer != 0`), validate, pack, in FORMAT CHECK + ZIP CHECK theo spec. Dùng để smoke-test format end-to-end trước khi nộp full.
- Cũng có (predate memory, unlisted trước): `build_retrieval_index.py`, `fix_table_ref_format.py`, `verify_table_filter.py`, `nrows_distribution.py`.
- `smoke_test.py` — M0 verify data read no API. `test_llm_api.py` — one-shot LLM API verify qwen3.5-9b.

### `docker/docker-compose.yml`
Qdrant `qdrant/qdrant:v1.19.0` (khớp qdrant-client 1.19.0), container `vifinqa-qdrant`, ports 6333/6334, volume `data/derived/qdrant`, restart unless-stopped.

### `tests/` (10 files)
`test_config.py` (deep-merge, extra=forbid), `test_loader.py` (100 stocks HPG→Hòa Phát), `test_numbers.py` (parse_vn_number/detect_unit/normalize_label/parse_period_header/unit_factor), `test_parser.py` (split_pages/extract_tables/grid colspan HPG 2018), `test_statements.py` (classify HPG/VCB/VJC), `test_catalog_builder.py` (header_and_labels skip section-title, unit_factor, merge), `test_facts.py` (M2 gold find_item_code_col/group/build_asset/emit_facts/validate_asset/build_report_facts), `test_entity.py` (9 câu golden Q1 VJC/Q4 FTS/Q11 BID/Q369 HPG+HSG+MSR+NKG/Q790 VIB+BID), `test_sandbox.py` (AST safety + run_pandas + `single_df_alias` + `bare_df_nameerror`), `test_codegen_prompt.py` (build_messages schema, assert `dfs["df1"]`).

## 3. Kiến trúc & luồng pipeline

### ETL flow (offline, toàn corpus — CẬP NHẬT 8/8-9/8)
```
OCR .txt → loader.ReportMeta
  → catalog_builder.process_report:
      parser.split_pages → parser.parse_table_grid (colspan + rowspan expand)
      → format_classify.classify_table (statement + TableLayout; notes/segment cũng detect layout)
      → write wide CSV derived/tables/{report_id}/table_{N}.csv (OCR strings preserved)
      → layouts/{rid}.json (mọi bảng) + CatalogRow (23 cols, gồm start_line + layout cols)
  → rebuild_evidence.py (MỚI 8/8): wide + layout → evidence/{rid}__table_N.csv [chi_tieu,Mãsố,ky,value] TOÀN BỘ 146,246 bảng
  → run_facts.py: facts_all.csv (400,510 rows, 12 cols)
  → run_merged_evidence.py: evidence_merged/ (5,474) + statement_meta.csv
  → documents.csv
```

### Retrieval flow (CẬP NHẬT 8/8-9/8)
```
question → RetrievalPipeline.search:
  1. extract_entities (entity.py): tickers (4-stage), years, report_type, statement hint (soft), unit factor
  2. build_payload_filter (search.py): Qdrant Filter must ticker/year/report_type
  3. embed_query: dense OpenRouter **nemotron-3-embed-1b:free 2048-d** (cache .npy per ticker); sparse tf_sparse (→ đã thay: bge-m3 1024-d local/DeepInfra, xem §0.1.7)
  4. hybrid_search: native Qdrant prefetch [dense, sparse] + FusionQuery(RRF) — nhưng **use_sparse=False (dense-only)**
  5. apply_statement_bonus: soft add if table.statement==hint, re-sort
  6. **fallback (MỚI 8/8)**: filter report_type quá hẹp (báo cáo 'other' không tách cons/sep — EVF/FTS) → bỏ filter type tìm lại
  7. top cfg.retrieval.k (**k=10**)
Qdrant bctc_tables: **118,728 points, dense 2048-dim** COSINE HNSW (INT8 quantize) + sparse IDF. Docker v1.19.0 port 6333/6334. **Payload index tường minh (year/ticker/report_type/statement) — fix filter AND trả 0.**
```

### Codegen + Sandbox flow (MỚI 8/8-9/8 — deterministic trước LLM)
```
solve(question, qid, pipeline, facts_index, llm, cfg, max_retries=1):
  1. results, entities = pipeline.search(question)
  2. _plan_evidence (MỚI 7/8): bảng statement → evidence_merged (gộp toàn bộ, preview 20k); notes → tidy per-table; dedupe (report_id, statement)
  3. **deterministic (MỚI 7/8 — engine/deterministic.py)**: solve_deterministic → lookup đơn giản, match label trong facts/evidence
       → nếu match: build_template_query → record _error="deterministic(facts|evidence)", KHÔNG gọi LLM
  4. nếu không match: build_messages → llm.generate_query (thinking=False)
  5. retry loop ≤2: _bad_refs (cấm dfs["..."], dùng bare df1/df2) → repair; check_code AST (8000/800) → repair; run_pandas (python -I, bare df1/df2 + dfs compat)
  6. fallback: answer=0.0
  7. _make_record → relevant_tables=report_id|table_N (NỘI BỘ), evidence variable=df{i+1} csv_path=data/{rid}__{tid}.csv
```

### Submission flow
```
results.jsonl → builder.build:
  load records, load_questions, find missing_ids
  _load_unit_factors (catalog_tables.csv)
  _load_start_lines (catalog_tables.csv cột start_line) → {(report_id, table_id): start_line}
  each evidence: parse (var, rid, tid) → _flat_name → copy tidy từ _source_tidy_path; thiếu → regenerate từ wide wide_csv_to_tidy(uf)+write_tidy_csv (rỗng→header-only) → data/{rid}__{tid}.csv
  fallback missing_ids (answer=0.0)
  sort by id, bỏ _ok/_error
  rewrite relevant_tables: rid|table_N → rid|<start_line> via _table_ref_to_line, count relevant_tables_rewritten
  _with_vn_num(pandas_query), ghi submission.json
→ validate: re-exec mỗi query trên CSV packaged (evidence keyed theo table_ref _table_ref_from_csv_path), so answer tol 0.01, short-circuit result=0.0, ThreadPoolExecutor workers=8, AST check (cfg 8000/800) → run_pandas → ok/crash/mismatch/bad_path
  → validation.jsonl + summary {total, ok, crash, mismatch, no_evidence, bad_path}
→ pack: submission.zip (json + data root, 1 json enforced, sort) + checksum.txt
```
**Lưu ý quan trọng:** Builder rewrite `relevant_tables` từ `rid|table_N` (internal) → `rid|<start_line>` (submission spec) ở bước packaging. Record nội bộ `results.jsonl` và grader dfs dict key vẫn dùng `rid|table_N`. Filename CSV vẫn `data/{rid}__{tid}.csv` (table_N). **Chỉ field `relevant_tables` scoring dùng line format.**

## 4. Schema dữ liệu & outputs

### `data/derived/` (ETL outputs — CẬP NHẬT 8/8-9/8)
- `tables/{report_id}/table_{N}.csv` — **146,246 wide tables** raw OCR strings (rowspan-aligned, dtype=str downstream). **CANONICAL**.
- `catalog_tables.csv` — **146,246 rows, 23 cols** (`report_id, ticker, year, report_type, table_id, page_no, unit, unit_factor, is_statement, statement, header_text, row_labels, n_rows, n_cols, anchor_context, start_line, header_idx, period_row_idx, code_col, label_col, period_cols, thuyet_minh_cols, number_format`). `is_statement=1`: 10,797 (BS 5,066 / CF 3,382 / IN 2,349); `is_statement=0`: 135,449 (notes). `period_cols` non-empty: **119,810** (82%); có `code_col`: 22,646. **`start_line`** = physical line của `<table>` trong OCR → builder emit `relevant_tables` as `rid|<start_line>`.
- `layouts/{rid}.json` — **1,973 file**, `{table_id: TableLayout}` mọi bảng (kể cả notes/segment).
- `facts_all.csv` — **400,510 rows, 12 cols** (`ticker, year, report_type, statement, item_code, item_label, item_label_raw, period_key, period_label, value_vnd, src_table_ids`). **Không có cột `report_id`** (deferred issue).
- `documents.csv` — **1,965 rows, 7 cols**.
- `evidence/` — **146,246 tidy CSV** (rebuilt 8/8, schema `[chi_tieu, Mãsố, ky, value]`) — mọi bảng kể cả notes.
- `evidence_merged/` — **5,474 statement gộp**. `statement_meta.csv` — 5,474 dòng.
- `embeddings/{TICKER}.npy`+`.json` — 100 tickers, dense **nemotron 2048-d**, cache key hash(texts)+n+model+dim. (→ đã thay: bge-m3 1024-d, xem §0.1.7)
- `qdrant/` — Docker Qdrant. Collection `bctc_tables`: **118,728 points, 2048-dim** (re-index 8/8, nemotron free, ~$0.27 cho bge-m3 trước). ⚠️ Mismatch 118,728 vs 146,246 — 27,518 bảng bị filter (min_n_rows=5/empty/n_rows==0). (→ đã thay: dim 1024, cần `--rebuild`, xem §0.1.7)
- State: `etl_state.json` (100 tickers), `facts_state.json` (100), `retrieval_state.json` (100, nemotron build 8/8).

### `data/out/` (codegen/results/submission)
- `results.jsonl` — 3,977,213 bytes, **1,012 records** (`id, question, answer, relevant_docs, relevant_tables, evidence, pandas_query`). Answer zero=640 / nonzero=372. Codegen full run 6/8 17:05: **ok=607, fail=405** (fallback 0.0). p95 latency 79s early → 120s late (throttling). Total wall 6,849s (~1.9h).
- `codegen_full.log`, `rebuild_index.log`, `_fts.txt`/`_fts2.txt` (diagnostic FTS 2023 tidy/dfs contract).
- `submission/submission.json` (**4,899,104 bytes**, 1,012 answers). `submission.zip` (4,875,351 bytes). `data/` (7,215 tidy CSV). `validation.jsonl`: **ok=968 / fail=44** (internal re-exec dfs contract; 361 extra ok là short-circuit `result=0.0` pass validator không spawn). `checksum.txt`.

### Submission record schema (packaged, sau builder rewrite)
```json
{
  "id": <int>,
  "question": "<string>",
  "answer": <float>,
  "relevant_docs": ["<report_id>"],
  "relevant_tables": ["<report_id>|<start_line>"],
  "evidence": [{"variable": "df1", "csv_path": "data/<report_id>__table_N.csv"}],
  "pandas_query": "<string>"
}
```
- `relevant_tables` = **`<report_id>|<start_line>`** (builder rewrite từ internal `rid|table_N` qua `_table_ref_to_line`, dùng cột `start_line` từ `catalog_tables.csv`). Đây là format OFFICIAL SPEC.
- `variable` = `df{i+1}` trong record, nhưng **grader dfs dict key theo `table_ref`** `report_id|table_N` (không phải theo variable; key match `csv_path` filename).
- `csv_path` filename vẫn dùng `table_N` (`data/{rid}__table_N.csv`) — filename là tuỳ chọn ta, chỉ `relevant_tables` VALUE là line format.
- `pandas_query` submitted = `_VN_NUM_DEF + "\n\n" + code` (builder nhúng def vn_num vì grader không inject).

## 5. Cấu hình & knobs

### `configs/base.yaml` (base — CẬP NHẬT 9/8)
- `paths`: data/data/data-out.
- `retrieval`: **k=10** (5→10, 9/8 — bảng đúng hay rank 6-8), rerank_depth=30, engine=qdrant, **use_dense=true, use_sparse=false** (9/8 — dense-only, sparse TF gây nhiễu tiếng Việt), fusion=native, statement_bonus, embedding max_chars=4000 (base; **api.yaml override 2000**), **workers=4** (base; **api.yaml override 12**), embed_statement_only=false, min_n_rows=5, rerank.enabled=false.
- `sandbox`: timeout=20, max_code_len=8000, max_ast_nodes=800.
- `llm`: openrouter, model_id=qwen/qwen3.5-9b, temperature=0.0, max_tokens=4096, timeout=60.0, retries=3 (defaults; api.yaml override). (→ đã thay: local_vllm.yaml = vLLM local + DeepInfra fallback, xem §0.1.7)

### `configs/api.yaml` (override, default cho mọi run — CẬP NHẬT 9/8)
- `llm`: max_tokens=4096, **timeout=30.0, retries=1**, extra_headers `HTTP-Referer`, `X-Title`.
- `retrieval.embedding` (MỚI 9/8): **provider=openrouter, model=`nvidia/nemotron-3-embed-1b:free`, dense_dim=2048, max_chars=2000, batch_size=100, workers=12, cache_dir=data/derived/embeddings**. (→ đã thay: local_vllm.yaml = bge_m3_server local + DeepInfra fallback, dense_dim=1024, xem §0.1.7)

### Knobs quan trọng (runtime)
| Knob | Value | Ý nghĩa |
|---|---|---|
| `llm.timeout` | 30.0s | per API call timeout (fix session 60→30) |
| `llm.retries` | 1 | SDK transport retry (fix session 3→1; config default 3) |
| `llm.max_tokens` | 4096 | Qwen3.5-9B output cap |
| `llm.thinking` | False | `extra_body={"reasoning":{"enabled":False}}` — tắt reasoning tokens Qwen3 (mặc định ăn 4258 tok → empty content) |
| `sandbox.timeout` | 20s | runner subprocess timeout |
| `sandbox.max_code_len` | 8000 | AST check code len (runtime; ast_check default 4000) |
| `sandbox.max_ast_nodes` | 800 | AST check node count (runtime; ast_check default 300) |
| `retrieval.k` | **10** | top-k final (9/8: 5→10) |
| `retrieval.use_dense` / `use_sparse` | true / **false** | dense-only (9/8) |
| `retrieval.embedding.model` | **nvidia/nemotron-3-embed-1b:free** | embed free, dim 2048 (9/8) |
| `retrieval.embedding.max_chars` | 2000 | chunk cắt (9/8) |
| `retrieval.embedding.batch_size/workers` | 100/12 | tốc độ embed (9/8) |
| `retrieval.embed_statement_only` | false | index full 146K (fix gap notes) |
| `retrieval.rerank.enabled` | false | Qwen3-Reranker tắt |
| `WALL_CAP` (run_codegen) | 120s | `cfg.sandbox.timeout * 6.0` per-question wall guard (session fix, CHƯA test kỹ — quá aggressive cho hard tail) |

**Model:** LLM `qwen/qwen3.5-9b` via OpenRouter. **Embeddings `nvidia/nemotron-3-embed-1b:free` 2048-d via OpenRouter (9/8)** — thay `baai/bge-m3` 1024-d. Qdrant Docker v1.19.0 port 6333/6334. Chưa có GPU local — rerank local bỏ. (→ đã thay toàn bộ: vLLM local LLM + bge_m3_server 1024-d embed + DeepInfra fallback chain; api.yaml chỉ còn fallback profile; xem §0.1.7)

## 6. Trạng thái hiện tại (milestones)

- **✅ M0 xong (4/8)** — scaffold, data load OK (1,012 câu / 100 ticker / 1,973 report / ~97,860 bảng estimate).
- **✅ M1 xong (4/8)** — ETL toàn corpus 146,246 wide + catalog 10,797 statement + documents. 11 code-review finding fix. 55 test pass.
- **✅ M2 xong (4/8)** — Facts tier 3 BCTC, 377,578 rows. Fix English number format (FPT/DBC/VGC `,`nghìn). 72 test pass.
- **✅ M3 xong (5/8)** — Retrieval hybrid Qdrant + entity extraction, **rerank TẮT**. 10,616 points (statement-only). 84 test pass.
- **✅ M4 xong (5/8)** — Text-to-Pandas codegen + sandbox + submission (gộp M4+M5simple+M6). Smoke 5 câu. 105 test pass.
- **✅ Fix B (6/8)** — sandbox contract refactor khớp grader BTC: runner inject `dfs` dict keyed theo `table_ref` + `df` alias, dtype=str/keep_default_na=False/index_col=None; prompt dạy `dfs["<table_ref>"]`; builder package tidy + regenerate fallback; `_extract_code` giữ strip `def vn_num` (builder inject `_VN_NUM_DEF`). 107 test pass. Codegen full run 6/8 17:05 đã có 1,012 records (ok=607/fail=405), submission validate ok=968/44.
- **✅ Fix format `relevant_tables` (session 6/8, code DONE — xem §7b)** — ETL ghi `start_line`, builder rewrite `relevant_tables` sang `report_id|<start_line>` per OFFICIAL SPEC. Code committed; **chưa rebuild submission + resubmit** để test TABLES_F2.
- **M5 ReAct đầy đủ** — đang dở (multi-tool inspect_table/get_facts/search_tables trong loop + self-consistency).
- **M7 dev-set 40 câu + metrics** — pending.

### Phiên 8/8-9/8 — refactor canonical + embed nemotron + rebuild toàn bộ

**Refactor kiến trúc (canonical wide + layout làm nguồn sự thật):**
- `parser.py` fix **rowspan** (header ngân hàng lệch cột → expand rowspan).
- MỚI `format_classify.py` (TableLayout) — mọi bảng map về layout chuẩn; catalog **16→23 cột**; MỚI `layouts/{rid}.json` (1,973 file).
- `catalog_builder.header_and_labels` dùng **`label_col` từ layout** (fix lấy nhầm cột STT "1|2|3" — giảm 4,215→12 bảng sai).
- `tidy.grid_to_tidy` dùng layout (label_col/code_col/period_cols/unit_factor) — không đoán cột.
- **Rebuild toàn bộ 8/8**: wide (736s) + evidence 146,246 (786s) + facts 400,510 (889s) + merged 5,474 (151s) + index.

**Embed model (9/8):** chốt **`nvidia/nemotron-3-embed-1b:free`** (OpenRouter, free, **dim 2048**). Lịch sử: bge-m3 → GPU thuê RTX 3060 (OOM 12GB, mạng chậm) → nemotron free (tốt hơn bge-m3 trong verify tiếng Việt). Fix: `encoding_format="float"` (nemotron không hỗ trợ base64). Index rebuild 9/8: 118,728 points, ~70 phút, free. (→ đã thay 15/8: quay lại bge-m3 1024-d local (bge_m3_server) + DeepInfra fallback, lý do OOM cũ không còn áp dụng khi serve riêng port 8000; rebuild bắt buộc vì đổi dim; xem §0.1.7)

**Qdrant fix 9/8:** payload index tường minh (year/ticker/report_type/statement) — filter AND (ticker+year) trả 0 khi không có index.

**Verify 30 câu deterministic (9/8, nemotron):** **14/30 match** (bge-m3 trước: 5/30). Q1/Q8/Q15/Q36 đúng.

**Codegen 20 câu (9/8):** 9 câu đúng (Q1,2,4,6,8,9,10,15,31), 8 câu 0.0 (notes phức tạp: Q3,7,12,18,19,25,36). 2 vấn đề retrieval tách rõ (xem §0.1.6): Lớp A (row_labels cắt 10 dòng — Q7), Lớp B (bảng % không period_cols → evidence rỗng — Q19).

**README.md + ARCHITECTURE.md (9/8):** viết lại đầy đủ — README = định dạng dữ liệu 7 tầng + vấn đề; ARCHITECTURE = luồng + module + config. Đọc trước khi làm việc.

### Lần nộp đầu (6/8) — leaderboard
`TABLES_F2=0.0, DOCS_F2=0.77, ANSWER_ACC=0.105, EXEC_ACC=0.020`. 3 vấn đề độc lập:
- **(A) TABLES_F2=0 — ĐÃ CÓ HƯỚNG FIX session 6/8 (code done, chờ resubmit).** Nguyên nhân likely **format SAI**: ta nộp `report_id|table_N`, nhưng OFFICIAL SPEC yêu cầu `report_id|<line>` (số dòng `<table>` trong OCR). Lần đầu CLAUDE.md/MEMORY tưởng `table_N` đúng (dựa companion repo `DSKT-NOWJ/ViFinQA` `make_table_ref`), nhưng spec BTC và example `|350` cho thấy là LINE NUMBER — "fix" 6/8 sang `table_N` là **hướng SAI**. **Fix lần này**: ETL thêm cột `start_line` (physical line của `<table>`), builder rewrite `relevant_tables` → `rid|<start_line>` ở packaging. ⚠️ **RISK**: spec example `|350` không khớp `<table>` line thật trong AAA 2015 (table_1 ở line 19, không 350) → example có thể illustrative HOẶC BTC đếm line khác (corpus nội bộ `ocr_filter/`). Không có gold để verify line-counting khớp BTC. Nhưng line-format rõ ràng literal spec, đúng hơn `table_N` (đã cho F2=0). **Nếu resubmit vẫn F2=0 → tự điều tra: thử các cách đếm line khác (0-based, tính cả dòng `<table>`, theo page) + đối chiếu `relevant_docs` đúng/sai để isolate retrieval vs numbering.** Fix này KHÔNG ảnh hưởng ANSWER/EXEC.
- **(B) EXEC_ACC=2% ≪ ANSWER_ACC=10.5% — BUG CONTRACT SANDBOX, fix B đã refactor.** Grader benchmark dùng dict `dfs` keyed theo `table_ref` (không inject bare df1/df2), CSV `dtype=str/keep_default_na=False/index_col=None`, không có helper `vn_num` (LLM parse thủ công/builder inject def). Query ta cũ dùng bare `df1`/`df2` + `.index.astype(str).str.contains` → NameError/KeyError ở grader → crash. **FIX B đã refactor xong 107 test** (runner dfs dict, prompt dfs["<table_ref>"], builder tidy + _VN_NUM_DEF). Codegen re-run 6/8 17:05 đã dùng contract mới (results.jsonl 1,012 records). Internal validate ok=968/44. **Kỳ vọng EXEC_ACC lên từ 2%** (chưa verify trên leaderboard).
- **(C) Retrieval gap — `embed_statement_only=true` bỏ 135K bảng notes.** Fix: re-index 146K (`embed_statement_only=false`). **Đã re-index 6/8 14:57**: 118,728 points (mismatch 27,518 — filter build_table_chunks min_n_rows=5/empty, không phải embed fail).

### Trạng thái data hiện tại (end session 6/8)
- Codegen full run 1,012 done (ok=607/fail=405, answer zero=640/nonzero=372).
- Submission packaged (cũ, `relevant_tables` chưa rewrite line format): submission.json + 7,215 tidy CSV + zip 4.88MB. Internal validate ok=968/fail=44.
- **Chưa nộp lại** với format line mới (cần rebuild submission via `build_submission.py`, xem §9).

## 7. Phiên 6/8/2026 — codegen speed/reliability

**Bối cảnh:** User chạy pipeline ViFinQA tại D:\GURU. Phiên tập trung vào tốc độ/độ tin cậy codegen run.

### INCIDENT 1 — Hai process python trùng lặp
- User lỡ tay launch `run_codegen.py` 2 lần lúc 10:25:22 (PID 2080 đang chạy thật, PID 21128 stuck zombie 31ms CPU/4MB RAM/never wrote records).
- Kiểm tra `results_tidy.jsonl`: 510 lines, 510 unique IDs, 0 duplicate (zombie không bao giờ write).
- Kill cả hai. 21128 đã tự gone, 2080 terminated.

### INCIDENT 2 — Codegen run degrade theo thời gian (VẤN ĐỀ THẬT)
- Run đầu (PID 2080, workers=8, timeout=60s, retries=3, manual max_retries=4, backoff `2^attempt`): hoàn thành 550/1012 trong ~66min.
  - p95 latency climbing 47s → 210s. fail rate climbing 11% → 36%.
  - Log cadence `[N/1012]` là print-only mỗi 10 completion, **không phải sync barrier** (ThreadPoolExecutor + as_completed = "finish → jump to next").
- **Root cause slowness: DOUBLE RETRY LAYER.**
  - Layer 1: OpenAI SDK `max_retries=3` (cfg.llm.retries) — transport-level retry với backoff riêng.
  - Layer 2: `generate_query()` manual loop `max_retries=4` (hardcoded llm.py:150) với `time.sleep(2^attempt + jitter)` = lên tới 15s sleep + 4×60s timeout calls.
  - Worst case ONE `generate_query` ~ 4×60 + 15 ~ 255s. `solve()` làm tới 3 LLM calls (1 initial + 2 repair, max_retries=2 trong loop.py) → có thể vượt 600s.
  - Giải thích p95=210s.

### 5 FIXES ÁP DỤNG (committed working tree, CHƯA re-run hoàn chỉnh)
1. `codegen/llm.py`: `generate_query` max_retries 4→2; backoff `2^attempt` → fixed `2.0s + jitter`.
2. `codegen/llm.py`: `OpenAI(max_retries=1)` hardcoded (was `llm.retries=3`) — kill double-retry, SDK làm 1 quick transport retry, manual loop là single retry layer.
3. `configs/api.yaml`: `llm.timeout` 60.0→30.0; `retries` 3→1 (consistent với hardcoded).
4. `agent/loop.py`: `solve()` max_retries default 2→1 (1 initial + 1 repair = 2 LLM calls max).
5. `scripts/run_codegen.py`: thêm `WALL_CAP = cfg.sandbox.timeout * 6.0 = 120s` per-question wall-clock guard via daemon thread + `queue.get(timeout=WALL_CAP)`. Timeout → fallback record `answer=0.0`, worker freed. Added imports `queue, threading`.

**Kỳ vọng sau fix:** 1 LLM call max ~2×30s + 2×2s ~ 64s; cả câu max ~140s nhưng wall-cap 120s cut → p95 ≤90s.

### RE-RUN RESULT (codegen_tidy2.log, đã bị user clean up)
`[10/462] ok=0 fail=10 p95=120.0`, `[40/462] ok=5 fail=35 p95=120.0`.
- **VẤN ĐỀ:** ok gần 0, ALL questions hitting wall-cap 120s.
- **Chẩn đoán:** API KHÔNG phải issue (single call 2.1s, 8-concurrent ~10s, không 429).
- 462 pending questions là **HARD TAIL** — run đầu submit all 1012, 550 fast completed, 462 slow vẫn in-flight khi kill. Các câu này thật sự cần >120s trong `solve()` (không phải LLM — có thể retrieval/tidy/card-build phases, HOẶC multiple LLM calls dưới load).
- **WALL_CAP=120s QUÁ AGGRESSIVE** cho hard questions — cut off trước khi `solve()` return cả fallback. Daemon thread **leak** (continues running) nhưng record written as timeout.
- **UNRESOLVED:** cần phase-time một pending question thật (search vs build_cards vs codegen) để tìm bottleneck thật. Bị ngắt trước khi hoàn thành diagnostic. `codegen_tidy2.log` và `results_tidy.jsonl` đã bị user clean up.

### KEY INSIGHT
Wall-cap approach đánh đổi correctness lấy speed trên hard questions. Fix tốt hơn: (a) raise wall-cap 180-240s, HOẶC (b) make `solve()` time-aware (check deadline giữa các phase, return fallback graceful thay vì daemon-thread leak), HOẶC (c) tìm why hard questions slow (retrieval? tidy? repeated LLM?).

### CURRENT STATE (end session)
- Config fixes vẫn trong `api.yaml` (timeout 30, retries 1). Code fixes vẫn trong `llm.py`, `loop.py`, `run_codegen.py` (**uncommitted**, git status shows M).
- `results_tidy.jsonl` deleted by user. `results.jsonl` (3.9MB, 17:05) tồn tại — likely earlier full run (wide or tidy contract).
- Không có python process running. Không có active codegen run.
- 107 test pass (per CLAUDE.md M4b note, trước session run_codegen wall-cap addition — wall-cap chưa test kỹ).

## 7b. Phiên 6/8/2026 — fix format `relevant_tables` sang `report_id|<line>`

**Motivation:** Lần nộp đầu `TABLES_F2=0.0`. OFFICIAL SPEC (BTC, authoritative 6/8) quy định `relevant_tables = "report_id|<vị trí bảng>"`, vị trí = **số dòng bắt đầu của bảng trong file OCR** (physical line). Spec example `AAA_financial_statements_2015_consolidated|350`. CLAUDE.md/MEMORY cũ tưởng `table_N` đúng (dựa companion repo `DSKT-NOWJ/ViFinQA` `make_table_ref` → `f"{doc_name}|table_{table_id}"`) — **SAI** theo spec BTC. "Fix" 6/8 sáng từ `report_id|{position}` → `report_id|table_{position}` là **hướng sai** (làm F2 vẫn 0). Companion repo dùng `table_N` nội bộ (từ `ocr_filter/*_extracted_tables/table_N.csv`) nhưng grader BTC leaderboard dùng LINE format.

**CHANGES (code DONE this session):**

A. **ETL `start_line`** (`src/vifinqa/etl/catalog_builder.py`):
- `CatalogRow` thêm field `start_line: int` (line 50).
- `table_start_lines(full_text: str) -> dict[int, int]` (line 64-75): trả `{table_idx (1-based, whole-report): physical line number của <table>}`, compute qua `TABLE_RE.finditer` + `full_text.count("\n", 0, m.start()) + 1`; fallback 0 nếu missing. Import `TABLE_RE` từ parser.
- `CATALOG_HEADER` thêm `"start_line"` làm cột 16 (line 22-26). `as_row()` append `str(start_line)`.
- `process_report` compute `start_lines = table_start_lines(text)` once/report (line 123), pass `start_line=start_lines.get(table_idx, 0)` vào mỗi `CatalogRow` (line 165).
- **Verified line-counting khớp `grep -n`**: AAA_financial_statements_2015_consolidated table_1→line 19, table_2→214 (BS assets), table_3→239, table_4→286 (income), table_5→333 (cash_flow), table_6→433. Mỗi `<table>` nằm trên 1 physical line trong OCR txt.

B. **Backfill script** (`scripts/backfill_start_line.py`, NEW) — tránh full ETL re-run:
- Đọc OCR của 1,973 report, compute `table_idx→line`, join vào `catalog_tables.csv` thêm cột `start_line`. **Không** re-write 146K wide CSV. Idempotent.
- **ĐÃ CHẠY**: 146,246/146,246 rows có `start_line != 0`, 0 report missing OCR. `catalog_tables.csv` nay 16 cols.

C. **Builder rewrite** (`src/vifinqa/submission/builder.py`):
- `_load_start_lines(derived_dir) -> {(report_id, table_id): start_line}` (line 94-110) đọc cột `start_line` từ `catalog_tables.csv`.
- `_table_ref_to_line(key, start_lines)` (line 113-127) convert `"report_id|table_N"` → `"report_id|<start_line>"`; fallback original nếu `start_line` missing/0.
- `build()` load `start_lines` (line 170), rewrite mỗi record `relevant_tables` (line 219-223), đếm `relevant_tables_rewritten`. `build()` return thêm `relevant_tables_rewritten` (cùng `n, materialized, tidy_regen, missing_ids`).
- `csv_path` **không đổi** (`data/{rid}__{tid}.csv`, filename giữ `table_N`).

D. **Smoke test** (`scripts/smoke_fmt_100.py`, NEW — **chưa chạy**, user interrupt trước): build 100-question subset (prefer `answer != 0`), validate, pack, in FORMAT CHECK + ZIP CHECK theo spec.

**NET EFFECT:** `relevant_tables` output = `report_id|<line>` (spec). `csv_path` filename vẫn `table_N`. **Không cần Qdrant re-index** (search stays `table_id` nội bộ). **Không cần codegen re-run** (`results.jsonl` giữ `table_N`; builder rewrite ở packaging). Để test TABLES_F2: rebuild submission + submit.

**⚠️ RISK:** spec example `|350` không khớp `<table>` line thật trong AAA 2015 (table_1 line 19, không 350) → example có thể illustrative HOẶC BTC đếm line khác (corpus nội bộ `ocr_filter/` không công khai). Không có gold verify line-counting khớp BTC. Nhưng line-format rõ ràng literal spec, đúng hơn `table_N` (F2=0). Nếu resubmit vẫn F2=0 → tự điều tra cách đếm line thay thế (xem §6 item A).

## 8. Ràng buộc cứng & gotchas

### Quy tắc mô hình (§5 CLAUDE.md)
- ✅ Chỉ LLM **open-source ≤14B**, phát hành **< 1/6/2026 (giờ VN)**. ✅ Qwen3.5-9B-Instruct (`Qwen/Qwen3.5-9B`) phát hành 2/3/2026 — hợp lệ.
- ❌ **CẤM** LLM đóng: GPT-4o, Gemini, Claude, ... (kể cả API). Cohere Rerank API = đóng → không hợp lệ. Jina Rerank = CC-BY-NC → rủi ro.
- Embedding/reranker cỡ nhỏ thường được chấp nhận (bge-m3 ~568M, Qwen3-Reranker-0.6B).

### Sandbox contract (post Fix 7/8/2026 — bare variables approach)
- Runner chạy `python -I` (isolated mode, drops PYTHONPATH/user site) → **không import vifinqa được**; `vn_num` phải inline copy.
- CSV read `dtype=str, keep_default_na=False, index_col=None, encoding=utf-8-sig` — cells = raw strings, empty=`''`, index numeric RangeIndex.
- **⚠️ PHÁT HIỆN 7/8/2026: Grader BTC inject BARE VARIABLES (df1, df2, ...) thay vì `dfs` dict!** Spec BTC nói rõ: "variable = Tên biến DataFrame đại diện cho bảng và **được sử dụng trực tiếp trong pandas_query**". Test thực tế: EXEC_ACC=0.004 với bare variables (tăng từ 0 với dfs dict). Companion repo `DSKT-NOWJ/ViFinQA` dùng `dfs` dict nhưng đó là code nội bộ tác giả, KHÔNG PHẢI grader BTC leaderboard.
- Runner inject: **bare variables `df1`, `df2`, ...** theo thứ tự evidence dict + giữ `dfs` dict cho tương thích + alias `df` khi 1 bảng.
- Namespace globals: `pd`, `vn_num`, `df1`, `df2`, ..., `dfs` (compat), `df` (alias 1 bảng), `print` (via `_safe_print`), `_SAFE_BUILTINS` (trừ open/eval/exec/compile/getattr/setattr/...). **No np/math/re/json** → NameError nếu dùng (mirror grader).
- AST check `_BLOCKED_CALLS` (open/eval/exec/read_csv/__import__/...), `_BLOCKED_MODULES` (os/sys/subprocess/socket/urllib/...), `_BLOCKED_ATTRS` (__class__/__globals__/__import__/...). `check_code()` function-param defaults = 4000/300, **nhưng runtime callers (agent.loop:177, build_submission→validate:63) pass `cfg.sandbox` values 8000/800 → effective limits = 8000/800**. `submission/validate.py:validate()` signature defaults = 6000/500 (cũng bị build_submission override bằng cfg 8000/800). ast_check docstring không nhắc 6000/500.

### Codegen contract (post 7/8/2026 — bare variables)
- Prompt dạy: **bare variables `df1`, `df2`, `df3`, ...** (theo evidence variable names). cells strings filter via boolean mask on column `.astype(str)`; **không** `.index`/`.iloc` by position; `vn_num` đã có sẵn (không redefine); end `result = round(<float>, 2)`.
- **⚠️ FIX 7/8/2026: Đổi từ `dfs["<table_ref>"]` sang bare `df1`, `df2`, ...** — do grader BTC inject bare variables (EXEC_ACC=0.004 với bare, 0.0 với dfs dict). Prompt mới: system/few-shot/user đều dùng `df1`, `df2`; validation `_bad_refs()` CẤM `dfs["..."]`; runner inject `df1`, `df2`, ... theo thứ tự evidence.
- `_extract_code` strip: think tokens → ```python fence (last) → imports → `def vn_num` (builder inject `_VN_NUM_DEF`) → `dfN = <synthetic>` reassign (giữ `df1 = df1[...]`).
- `thinking=False` via `extra_body={"reasoning":{"enabled":False}}` — Qwen3 reasoning mặc định ăn max_tokens (seen 4258 tok → empty content).
- Retry: SDK `max_retries=1` (hardcoded post-fix), manual loop `max_retries=2` (range(max_retries) = 2 total attempts: 1 initial + 1 retry) sleep `2.0+random(0,1)` trên timeout/connection + HTTP {429,500,502,503,504}.

### Submission format gotchas
- **`relevant_tables` = `report_id|<start_line>`** (OFFICIAL SPEC, builder rewrite từ internal `rid|table_N` qua `_table_ref_to_line` dùng `catalog_tables.csv.start_line`). ⚠️ "Fix" 6/8 sáng từ `report_id|{position}` → `report_id|table_N` là **hướng SAI** — companion repo dùng `table_N` nội bộ nhưng grader BTC dùng LINE format. Session 6/8 chiều đã sửa sang line format (code done, chờ resubmit).
- `csv_path` bắt đầu `data/`, file flat `data/{report_id}__{table_id}.csv` (filename vẫn `table_N` — filename là tuỳ chọn, chỉ `relevant_tables` VALUE là line format). `variable`=`df{i+1}` trong record, nhưng **grader dfs dict key theo `table_ref`** `rid|table_N` (match filename), không theo variable.
- `pandas_query` submitted = `_VN_NUM_DEF + "\n\n" + code` (grader không inject vn_num).
- Builder regenerate tidy từ wide nếu tidy thiếu (stale evidence), rỗng→header-only (query→0.0). Builder rewrite `relevant_tables` sang line format ở packaging (không thay record nội bộ `results.jsonl`).
- Validate short-circuit `result = 0.0` không spawn subprocess.
- Đúng 1 file .json ở root ZIP, không bọc thư mục cha.
- ⚠️ **MÂU THUẪN CONTRACT CHƯA RESOLVE (spec BTC vs Fix B — phát hiện 6/8/2026 khi đối chiếu spec lần cuối):** Spec ví dụ `pandas_query` dùng **bare `df1`** và định nghĩa `variable` = "Tên biến DataFrame đại diện cho bảng và **được sử dụng trực tiếp trong pandas_query**" → gợi ý grader BTC **inject DataFrame theo tên biến evidence (df1/df2/...)**. Fix B (dựa companion repo `answering/sandbox.py`) assume grader inject **dict `dfs` keyed theo table_ref**, cấm bare df1 → toàn bộ 1,012 query hiện tại dùng `dfs["<table_ref>"]`. **Nếu grader BTC theo đúng spec ví dụ → mọi query của ta NameError (`dfs` undefined) → EXEC_ACC ≈ 0, tệ hơn cả lần nộp đầu.** FIX đề xuất (**dual-compat, làm trong builder ở packaging, KHÔNG cần re-codegen**): với mỗi record, prepend shim `try: dfN\nexcept NameError: dfN = dfs["<table_ref>"]` cho mỗi evidence variable (table_ref lấy từ csv_path filename), rồi replace `dfs["<table_ref>"]` trong code body → bare `dfN`. Chạy được CẢ hai contract: grader inject bare → try pass (no-op); grader inject dfs dict → except bind từ dfs. Internal runner/validate cũng phải accept shim này (nó dùng try/except NameError — AST check hiện không chặn try/except, cần verify).

### Checkpoint / run gotchas
- Checkpoint per id trong `results.jsonl` — **không delete results.jsonl mid-run** (sẽ mất checkpoint, re-run từ đầu). `_load_done` parse existing ids skip done.
- `lock_file` defined trong run_codegen.py nhưng **unimplemented** — dùng plain append (race risk nếu 2 process cùng write, per INCIDENT 1).
- `WALL_CAP` daemon thread **leak** — continue running sau timeout, record written nhưng thread không join. Cần fix time-aware `solve()`.
- `etl_state.json`/`facts_state.json`/`retrieval_state.json` mark ticker done=1 — resume skip, không re-process. Error trong 1 ticker vẫn mark done (no brick) nhưng error log `*_errors.tsv`.

### Qdrant / retrieval gotchas
- Qdrant Docker `qdrant/qdrant:v1.19.0` port 6333 (REST+gRPC) / 6334 (dashboard). `mode=server` trong base.yaml. (Session verify: Docker offline WinError 10061 — không query live.)
- `point_id` uuid5 fixed namespace → idempotent (resume re-upsert, no duplicate).
- Cache `.npy` key = `hash(texts)+n+model+dim` — switch model invalidate cache.
- `embed_statement_only=false` (M4) — index full 146K. Re-index 6/8: 118,728 points (mismatch 27,518 — filter `build_table_chunks` min_n_rows=5/empty, không phải embed fail; số từ log, không query live).
- `facts_all.csv` **không có cột `report_id`** — không phân biệt HDB `_separate` vs `_separate_1` (M3 deferred).
- `facts_for_table` exact match `src_table_ids == table_id` (không substring — table_1 ≠ table_10).
- Statement bonus **soft** (0.001) — notes table_50 vẫn rank on content, không hard filter.
- RRF native Qdrant `FusionQuery(Fusion.RRF)` fusing dense+sparse prefetch trong 1 `query_points` call (không post-hoc merge).
- Reranker lazy-import (serve không cần torch). `rerank.enabled=false` — Qwen3-Reranker-0.6B CPU quá nặng (đã xoá model 1.2GB). `_DTYPE_KWARG` defined nhưng không apply trong `from_pretrained` (bug, không impact vì off).

### ETL gotchas
- `parse_vn_number("4.037") == 4037` (EPS — thousands sep, not decimal). `_PERIOD_LABEL_RE` excludes `Số năm` (depreciation notes). `detect_number_format` returns `vi` on ties.
- `parse_table_grid` **colspan expand** (text duplicate) nhưng **rowspan NOT handled** (accepted M1 wide tier).
- `detect_unit` anchor-bounded (không whole page — tránh 2nd table inherit 1st table unit).
- `_has_statement_structure` fallback **≥50% code col** + ≥3 data rows (không phải ≥70% — ≥70% là `find_item_code_col`). `find_item_code_col` requires ≥3 data rows. `build_asset` assumes fragments share structure (validated by `header_signature`).
- `emit_facts` dedupe item_code ở biên fragment (`seen_codes`), label inheritance cho blank label. `period_key` year-less → empty string.
- `tidy.wide_to_tidy` **parse VN only** — English BCTC (FPT) cần unit_factor pre-scaled. Opening-balance columns deliberately dropped. Code column = first match only.
- `table_start_lines` (session 6/8): mỗi `<table>` nằm trên 1 physical line; `full_text.count("\n", 0, m.start()) + 1` = line number 1-based. Fallback 0 nếu `TABLE_RE` no match.

## 8b. Lịch sử nộp bài và kết quả (7/8/2026)

### Các lần nộp và kết quả leaderboard

| Lần nộp | Ngày | Số câu real | Contract | TABLES_F2 | DOCS_F2 | ANSWER_ACC | EXEC_ACC | Ghi chú |
|---|---|---|---|---|---|---|---|---|
| **1** | 6/8 | 1012 (full) | `dfs["<table_ref>"]` | 0.0 | 0.77 | 0.105 | 0.020 | Format `table_N`, grader inject `dfs` dict? |
| **2** | 7/8 sáng | 100 + 912 fallback | `dfs["<table_ref>"]` | 0.0166 | 0.0847 | 0.0079 | 0.0 | Line format, nhưng 912 fallback kéo điểm |
| **3** | 7/8 trưa | 397 + 615 fallback | `dfs["<table_ref>"]` | 0.0668 | 0.3275 | 0.0474 | 0.0 | Line format, cải thiện retrieval |
| **4** | 7/8 chiều | 10 + 1002 fallback | **bare `df1`, `df2`** | 0.0021 | 0.0096 | 0.004 | **0.004** ✅ | **EXEC_ACC > 0! Bare variables ĐÚNG** |

### Bài học rút ra

1. **Grader BTC inject BARE VARIABLES (df1, df2, ...)** — KHÔNG PHẢI `dfs` dict keyed theo table_ref. Bằng chứng: EXEC_ACC tăng từ 0 → 0.004 khi đổi sang bare variables. Companion repo `DSKT-NOWJ/ViFinQA` dùng `dfs` dict nhưng đó là code nội bộ tác giả, KHÔNG phải grader BTC leaderboard.

2. **✅ BTC XÁC NHẬN (9/8): PHẢI NỘP ĐỦ 1012 CÂU** — mọi ghi chú cũ "gold=506, nộp subset" là SAI. Warning "gold=506 pred=1012" KHÔNG có nghĩa là chỉ cần nộp 506 — BTC yêu cầu đủ 1012 question. Nguyên nhân điểm thấp lần nộp 100 câu (DOCS_F2 0.7486 → 0.08) là do **912 câu fallback 0.0 kéo trung bình xuống**, KHÔNG phải vì nộp thừa. → Phải chạy đủ 1012 câu, tối thiểu hoá fallback.

3. **relevant_tables line format có match MỘT PHẦN** — TABLES_F2 tăng 0 → 0.0166 → 0.0668 qua các lần nộp với line format. Nhưng vẫn thấp, có thể do: (a) BTC dùng corpus nội bộ `ocr_filter/` khác với HF, (b) cách đếm line khác, (c) retrieval chưa chính xác.

4. **Contract bare variables đã resolve** — Prompt dạy `df1`, `df2`, ...; runner inject `df1`, `df2`, ...; validation cấm `dfs["..."]`. 100 câu test: 83/100 OK (83%), validate 91.7%.

### Việc cần làm

1. **Chạy đủ 1012 câu** (BTC xác nhận 9/8: phải nộp đủ 1012, KHÔNG subset) — tối thiểu hoá fallback 0.0 vì fallback kéo trung bình xuống.
2. **Cải thiện retrieval** — TABLES_F2 vẫn thấp. Tự tune retrieval (top-k, statement bonus, min_n_rows, entity filter) + đối chiếu `relevant_docs` đúng/sai để tách lỗi retrieval vs lỗi numbering.

## 8b. Phiên 7/8/2026 — merged evidence tier + deterministic engine (không LLM)

**Bối cảnh:** LLM codegen chậm (~30-60s/câu) + không ổn định (variance, hallucinate). Mục tiêu phiên: bỏ LLM cho lookup đơn giản bằng **deterministic engine** trên dữ liệu sạch, giữ LLM chỉ cho câu phức tạp. Đồng thời fix fragment-split (CF tách table_9+table_10).

### ✅ 1. Merged evidence tier (bảng gộp statement)
- `src/vifinqa/etl/merged_evidence.py`: `merge_statement_facts(report_year, facts_df)` reuse M2 `group_statement_fragments`/`build_asset`/`emit_facts` → mỗi BS/IN/CF **1 bảng gộp** schema `[chi_tieu, Mãsố, ky, value]` (label ASCII sạch từ `item_label`, value_vnd VND). Bỏ `period_key=year_start` (đầu kỳ), bỏ label rỗng + noise "Quyết định", dedupe (Mãsố, ky), ky từ period_label ∈ {ry,ry-1,ry-2} else ry.
- `scripts/run_merged_evidence.py`: `data/derived/evidence_merged/{rid}__{statement}.csv` (5,453 bảng gộp, 1,902 report) + `data/derived/statement_meta.csv` (report_id, statement, src_table_ids JSON).
- `agent/loop.py`: `_plan_evidence()` — bảng statement (trong statement_meta) → dùng **bảng gộp** (dedupe theo report+statement), notes → tidy per-table. `_facts_var()` map facts row → var dfN. Fix fragment-split (id 6 VSC CF: mã 20 = 145.73 tỷ ✓).
- `etl/tidy.py`: `wide_to_tidy` normalize `chi_tieu` → ASCII (đồng bộ toàn evidence; đã bulk regenerate 7,715 tidy CSV qua `data/out/_regen_evidence.py`).

### ✅ 2. Deterministic engine (không LLM, ~1s/câu)
- `src/vifinqa/engine/deterministic.py`: `solve_deterministic(question, entities, facts_index, evidence_paths, evidence_plan)`.
  - **Facts tier primary** (statement hint + label ASCII sạch + Mãsố bonus) → answer = value_vnd/unit_factor.
  - **Evidence tier fallback** (notes — mọi bảng retrieval trả), KHÔNG bonus mã.
  - `is_complex()` bỏ qua câu có so sánh/tăng trưởng/tỷ lệ/argmax → fallback LLM. Ngưỡng `_SCORE_MIN=0.5`.
  - `build_template_query()` sinh pandas_query deterministic (filter Mãsố nếu có, else chi_tieu contains ASCII) — re-exec khớp answer ✓ (verify 3 câu).
- `agent/loop.py solve()`: deterministic chạy **trước** LLM; nếu có → record `_error="deterministic(facts|evidence)"`, không gọi LLM.
- **Precheck sửa**: `_precheck_evidence` (hard-block 487/1012 câu → hồi quy nặng) → đổi thành `_precheck_hint` **advisory** (chỉ thêm hint vào prompt, không block).

### Kết quả 20 câu (test20b)
- 10/20 deterministic (5 facts + 5 evidence) — nhanh + kiên định (không LLM variance).
- 19/20 answer nonzero. id 1/3/4/6 đúng. id 8 (FTS lương) retrieval-fail. id 2/13/17/19 LLM trả 0.0 (cần retrieval/notes tune).
- So baseline: trước 48% precheck-block + 14% facts coverage → giờ ~50% deterministic + không block.

### Sinh full 1012 (results_det.jsonl, trước fix precheck)
- 118 deterministic / 487 precheck-block (đã fix sau) / 451 ok / 561 fallback. **Sau fix precheck cần re-run đầy đủ** (chưa chạy lại full — user yêu cầu test 20 câu trước).

### Test
- `tests/test_merged_evidence.py` (7 test), `tests/test_deterministic.py` (7 test). Full suite: **124 passed, 2 pre-existing fail** (test_catalog_builder start_line stale, test_config rerank model — không liên quan).

### ⚠️ Cần làm
1. Re-run codegen full 1012 với precheck fix (+ deterministic) → build submission → nộp đo EXEC/ANSWER.
2. Tune evidence tier cho notes segment (id 2 ACB "ngành Thương mại", id 8 FTS lương) — retrieval chưa trả đúng bảng segment.
3. LLM vẫn fallback cho câu complex (so sánh/tỷ lệ/argmax ~489 câu) — cân nhắc template arithmetic.

## 9. Việc cần làm kế tiếp (cập nhật 9/8/2026 — sau refactor canonical + nemotron)

**⚠️ Trước tiên — các vấn đề retrieval đã phân tích (xem §0.1.6), fix theo thứ tự:**

1. **✅ ĐÃ XONG (9/8): Fix Vấn đề #5 (Lớp B)** — `format_classify.value_col` cho bảng không-cột-kỳ + `tidy.grid_to_tidy` emit ky=report_year. Rebuild layouts/catalog/evidence/label_index. Q19 (HHV table_1 5 rows), Q20 (GVR Visorutex 27.78) evidence OK.
2. **✅ ĐÃ XONG (9/8): Fix Vấn đề #1 (Lớp A): lexical recall** — `retrieval/label_index.py` (inverted index chi_tieu → rid|table_id, 113,289 entry, pkl 44MB), `_label_recall` union bảng trong `solve()`. Q7 → merged BS có quỹ khen thưởng.
3. **✅ ĐÃ XONG (9/8): verify full 1012 (no-LLM)** — coverage 100%, deterministic match 219/1012 (21.6%), lexical thêm bảng 98% câu. Codegen 100 câu: ok=98/100, deterministic 63.
4. **Anchor noise filter** (`index.py::build_table_chunks`) — lọc dòng toàn hoa + không số, `TRANG|MỤC LỤC|KIỂM TOÁN`. ⚠️ Cần rebuild index ~70p.
5. **Deterministic mở rộng** (`engine/deterministic.py`) — intent classifier (tăng/giảm/tỷ lệ/argmax) + `difflib.SequenceMatcher` thay token-overlap + alias dict.
6. **Chạy codegen full 1012** (nemotron + lexical + fix #5) → build submission → nộp. **BTC xác nhận 9/8: PHẢI NỘP ĐỦ 1012 — không subset, tối thiểu hoá fallback.**
7. **Phase-time `solve()` trên hard question** — đo search vs build_cards vs codegen (bottleneck còn chưa rõ).
8. **M7 dev-set 40 câu + metrics.**
9. **Điều tra mismatch 118,728 vs 146,246** (27,518 bảng filter min_n_rows/empty).
10. **Working notes paper** (bắt buộc để kết quả chính thức).

## 10. Nguồn

- `D:\GURU\README.md` — **định dạng dữ liệu 7 tầng** + vấn đề (cập nhật 9/8).
- `D:\GURU\ARCHITECTURE.md` — **kiến trúc & luồng** (cập nhật 9/8).
- `D:\GURU\CLAUDE.md` — tài liệu dài hạn dự án (quy định cuộc thi/dữ liệu/phương pháp/kế hoạch).
- `D:\GURU\docs\plan_agentic_rag.md` — kế hoạch chi tiết cấp module, milestone, schema, rủi ro.
- Dataset: https://huggingface.co/datasets/AIGuruTinix/ViFinQA
- Companion repo: https://github.com/DSKT-NOWJ/ViFinQA (code generation, retrieval, reranking, answering, evaluation, config YAML; dùng `table_N` nội bộ từ `ocr_filter/*_extracted_tables/table_N.csv`). Paper *"ViFinQA: A Comprehensive and Challenging Benchmark for End-to-End Vietnamese Financial Reasoning."*
- Dashboard nộp bài: http://leaderboard.aiguru.com.vn/ → My Submissions.
- Grader benchmark reference: `DSKT-NOWJ/ViFinQA` `answering/sandbox.py` + `prompts/answering/program_system.txt` (dict `dfs` keyed theo `table_ref`, dtype=str/keep_default_na=False/index_col=None, no vn_num, `def` allowed, result scalar round 2).
- **OFFICIAL BTC SUBMISSION SPEC** (authoritative, user 6/8/2026): `relevant_tables = "report_id|<physical_line_number_of_table_in_OCR>"` (VD `AAA_financial_statements_2015_consolidated|350`), KHÔNG phải `report_id|table_N`.