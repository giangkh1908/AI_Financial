# Kế hoạch triển khai: Agentic RAG cho Text-to-Pandas trên BCTC (ViFinQA)

> Plan chi tiết theo từng mục/milestone. Bản chính thức của dự án — cập nhật khi có thay đổi.
> Mỗi milestone có: **Task** (việc làm), **File** (tạo ra), **Schema** (định nghĩa), **DoD** (Definition of Done — tiêu chí xong).

---

## 0. Context

Cuộc thi ViFinQA: với mỗi câu hỏi tài chính tiếng Việt (1,012 câu), xác định đúng bảng BCTC (table retrieval) + sinh pandas query thực thi được (text-to-pandas), nộp `submission.json` + thư mục `data/` (CSV evidence) → ZIP. **Grader chạy lại `pandas_query` trên đúng CSV ta nộp và kiểm tra `answer`** → tính tái lập là ràng buộc cấu trúc.

**Grounding đã khảo sát:**
- Dữ liệu: `data/code_stock.csv` (100 mã), `data/questions/questions.jsonl` (1,012 câu, không đáp án), `data/financial_statements/{TICKER}/{YEAR}/{report_id}/{report_id}_extracted.txt` (~100 ticker × 2015–2025 × consolidated/separate). OCR có `===== PAGE N =====` + bảng HTML `<table><tr><td>`, lỗi dấu nhiều.
- Bảng: cột đa năm, dòng section `colspan`, cột `Mã số` (VAS số 01–70/100–500; **ngân hàng mã La Mã `I,II,VI`**), unit trong dòng `Đơn vị: VND` hoặc ngay trong header (`31/12/2022Triệu VND`), số dấu chấm nghìn / âm trong ngoặc / `-`.
- Bảng bị cắt trang → phải **merge fragment** cùng schema (KQKD 2 table, LCTT 3 table). Bảng thuyết minh KHÔNG có item_code ("Lãi tiền gửi" VJC nằm ở bảng Thu nhập tài chính) → wide raw tier bắt buộc.
- Số table/report max quan sát = 248 (OCB 2025) → ví dụ BTC `|350` **không phải** ordinal per-report (xem R1).
- Câu hỏi: 472 câu 1 ticker, **364 câu không ticker** (chỉ tên công ty), 165 câu 3+ ticker; ticker viết thường; câu hard đa bước (ROE tại năm CFO/DT cao nhất giữa nhiều công ty); đơn vị triệu/trăm tỷ/nghìn tỷ đồng.

**Quyết định đã chốt:**
- LLM **Qwen3.5-9B-Instruct** (`Qwen/Qwen3.5-9B`; Apache 2.0, 9B ≤14B, phát hành 2/3/2026 < 1/6/2026 → hợp lệ; hybrid Gated DeltaNet + Gated Attention, 262K context, tool calling). Giai đoạn đầu dùng **API OpenRouter** — model ID `qwen/qwen3.5-9b` (đã xác minh có trên OpenRouter 4/8/2026, $0.10/$0.15 per 1M, 262K ctx), sau thuê GPU chạy local vLLM cùng model — client provider-agnostic.
- Agent: **custom ReAct** (JSON action text-based), không LangGraph/LlamaIndex.
- **BẮT BUỘC sandbox** chạy pandas_query (subprocess + hạn chế builtins/import + chặn network/file-write + timeout), dùng chung cho tool `run_pandas` và validator đóng gói.
- Embedding BGE-M3 + reranker bge-reranker-v2-m3 (CPU local). 2 tầng evidence: facts long-format + wide raw.

---

## 1. Kiến trúc tổng thể

```
data/financial_statements/**/_extracted.txt  (input read-only)
  │  ETL offline (1 lần, checkpoint theo ticker)
  ▼
parser (PAGE split + table grid) → statements (classify + item_code + merge fragment)
  → facts/{report_id}_facts.csv (long, VND) + tables/{report_id}/table_{N}.csv (wide raw)
  → catalog_tables.csv + documents.csv + facts_all.csv (~480K dòng)
  │  Retrieval index (offline)
  ▼
entity (ticker/company/years/report_type/unit) → BM25 + BGE-M3 dense → RRF → rerank → top-k
  │  Agent (per question, online) — custom ReAct, Qwen3.5-9B
  ▼
tools: search_tables | get_facts | inspect_table | run_pandas | finalize
self-correction ≤2 retries; unit conversion; validation feedback
  │  Submission (offline)
  ▼
builder (submission.json + data/*.csv) → validate (re-exec mọi query) → pack ZIP
```

**2 tầng evidence:**
- **Tier A — facts long-format** (`{report_id}_facts.csv`): 3 BCTC lõi, có `item_code`, chuẩn VND → bền với OCR noise, workhorse cho ratio/đa năm/đa công ty/argmax.
- **Tier B — wide raw** (`tables/{report_id}/table_{N}.csv`): mọi bảng còn lại (notes), giữ chuỗi OCR → cho chỉ tiêu không mã.

---

## 2. Module layout

```
D:\GURU\
├── pyproject.toml / requirements.txt   # pandas, numpy, beautifulsoup4, lxml, rank-bm25,
│                                       # sentence-transformers, FlagEmbedding, faiss-cpu,
│                                       # httpx, openai, pydantic, pydantic-settings, pyyaml, tqdm, pytest
├── configs/
│   ├── base.yaml       # paths, retrieval{k=10, rerank_depth=100}, sandbox{timeout=20}, tolerance=0.01
│   ├── api.yaml        # provider=openrouter, base_url=https://openrouter.ai/api/v1, model=qwen/qwen3.5-9b
│   └── local_vllm.yaml # provider=vllm, base_url=http://localhost:8000/v1
├── src/vifinqa/
│   ├── config.py  constants.py  loader.py
│   ├── etl/       parser.py  numbers.py  statements.py  facts_builder.py  catalog_builder.py  run.py
│   ├── retrieval/ entity.py  corpus.py  bm25.py  dense.py  rrf.py  rerank.py  facts_index.py  pipeline.py
│   ├── agent/     llm.py  tools.py  react.py  prompts.py  units.py  validation.py
│   ├── sandbox/   ast_check.py  executor.py  paths.py
│   ├── submission/ builder.py  validate.py  pack.py
│   └── eval/      devset.py  metrics.py  runner.py
├── scripts/      run_etl.py  build_retrieval_index.py  run_batch.py  label_dev.py  validate_submission.py
├── tests/        test_numbers.py  test_parser.py  test_statements.py  test_sandbox.py
│                 test_entity.py  test_submission_roundtrip.py
└── data/derived/ + data/out/   # gitignored (nằm trong data/)
```

---

## 3. Schema dữ liệu (contract giữa các module)

### 3.1 Facts — `data/derived/facts/{report_id}_facts.csv`
| Cột | Ý nghĩa |
|---|---|
| `ticker` | Mã CK (HPG, VCB...) |
| `year` | Năm báo cáo |
| `report_type` | `consolidated` \| `separate` |
| `statement` | `balance_sheet` \| `income` \| `cash_flow` |
| `item_code` | Mã số VAS (`110`,`411a`) hoặc bank (`I`,`VI`,`A`); `""` nếu không mã |
| `item_label` | Nhãn chuẩn hoá (diacritic-insensitive) |
| `item_label_raw` | Nhãn OCR gốc |
| `period_key` | `year_start` \| `year_end` \| `flow_year` \| `restated_cur` \| `restated_prev` |
| `period_label` | Nhãn kỳ gốc (`31/12/2018`) |
| `value_vnd` | **float chuẩn về VND** |
| `src_table_ids` | `table_N` đóng góp (→ `relevant_tables`) |

### 3.2 Wide raw — `data/derived/tables/{report_id}/table_{N}.csv`
Grid giữ nguyên chuỗi OCR (header multi-row nối dọc theo cột), **KHÔNG parse số**. `N` = số thứ tự `<table>` trong report (1-based).

### 3.3 Catalog — `data/derived/catalog_tables.csv`
`report_id, ticker, year, report_type, table_id, page_no, unit, is_statement, statement, header_text, row_labels, n_rows, n_cols, anchor_context` (anchor ≈ 6 dòng text trước bảng — chứa tên mục/đơn vị).

### 3.4 Documents — `data/derived/documents.csv`
`report_id, ticker, year, report_type, company_name, has_consolidated, has_separate`

### 3.5 Evidence per-question (trong ZIP `data/`)
- `data/facts__{qid}__{subset_key}.csv` — subset facts_all cho đúng ticker/stmt/năm (vài trăm dòng).
- `data/table__{report_id}__{N}.csv` — bảng wide dùng.
- `evidence = [{"variable":"df1","csv_path":"data/..."}, ...]`; 1 bảng → biến `df`, nhiều → `dfs['df1']`.

### 3.6 `submission.json`
1,012 record `{id, question, answer(float), relevant_docs[], relevant_tables[], evidence[], pandas_query}`. `pandas_query`: chỉ dùng `pd/np/math/re/json` + builtin allowlist, **không import**, gán `result`.

---

## 4. Chi tiết theo milestone

### M0 — Scaffolding & nền tảng (0.5 ngày)
**Task:**
1. Tạo venv + `requirements.txt` (đúng danh sách deps ở §2) + `pyproject.toml` (src layout, pytest).
2. `configs/base.yaml`, `api.yaml`, `local_vllm.yaml`.
3. `src/vifinqa/constants.py`: `UNIT_FACTORS={"nghìn":1e3,"triệu":1e6,"tỷ":1e9,"đồng":1,"VND":1}`, `ANSWER_ABS_TOL=0.01`, `STEP_BUDGET=10`, `SANDBOX_TIMEOUT=20`, `MAX_CODE_LEN=4000`, `MAX_AST_NODES=300`, `K=10`, `RERANK_DEPTH=100`.
4. `src/vifinqa/config.py`: `Config`, `LLMConfig(provider, base_url, api_key, model_id)` — provider-agnostic.
5. `src/vifinqa/loader.py`: `load_stocks() -> dict[ticker, name]`, `load_questions() -> list[dict]`, `iter_reports() -> list[ReportMeta(report_id,ticker,year,report_type,path)]`.
6. Smoke test: in 3 câu hỏi + 1 report path + đếm sơ bộ.

**DoD:** `pytest tests/` chạy được (skeleton); smoke in đúng 3 câu, 1 path report, số bảng sơ bộ; model Qwen3.5-9B gọi thử qua API 1 lần thành công (xác minh provider có serve model này — nếu chưa có, báo user để chọn provider khác hoặc fallback Qwen3-8B).

### M1 — ETL: parser + numbers + wide tables + catalog (2–3 ngày)

**Khảo sát OCR (4/8/2026 — HPG 2018 con, VCB 2022 con, VJC 2018 separate):**
- `<table>` luôn nằm gọn **1 dòng** (regex `&lt;table&gt;(.*?)&lt;/table&gt;` re.S). HTML-escape cần `html.unescape` (`&#x27;` → `'`).
- Header cột **chứa đơn vị nhúng vào cell năm**: `31/12/2018 VND`, `2018 VND`, `2018VND` (không space), `31/12/2022Triệu VND` (VCB), `31/12/2018Giá gốc/...VND` (VJC notes). → regex unit ưu tiên `(Nghìn|Triệu|Tỷ)?\s*VND` sau cell năm; fallback dòng `Đơn vị tính:`/`ĐVT:` (4434× VND, 1088× Triệu đồng VN, 607× Đồng VN, 16× tỷ đồng; OCR lỗi "Triệu Đông").
- CĐKT tách **2–3 `<table>`** (HPG 3, VJC 2), KQKD 1–2 (HPG 2, **row 60 lặp ở 2 fragment** — dedupe theo item_code chỉ ở biên fragment), LCTT 2 (section `colspan` "LƯU CHUYỂN...").
- Cột mã: HPG/VJC `Mã số` (VAS `\d{1,3}[a-z]?`, có `411a`, `421b`), VCB `STT` (La Mã `A,I,II,...,XII` + số con `1,2,3`). Row label continuation: label trống → mã ở row sau. Row `colspan=N` = section title (TÀI SẢN/NGUỒN VỐN/...), bỏ khi xây facts.
- `-` = giá trị rỗng/0; `(x)` = âm; `.` = phân cách nghìn; công thức LaTeX trong label `(\( 100 = 110 + ... \))` (giữ nguyên label, không parse).
- Gold test khớp: HPG LNST 60 = `8.600.550.706.227`; Tổng TS(270) = Tổng NV(440) = `78.223.007.670.925`; VJC "Lãi tiền gửi" (Thu nhập tài chính, bảng notes không mã) = `208.253.201.298` — bảng notes chỉ header `2018VND | 2017VND` → wide tier.


**Task & chữ ký hàm:**
- `etl/parser.py`:
  - `split_pages(text) -> list[Page]` — regex `===== PAGE (\d+) =====`.
  - `extract_tables(page_text) -> list[RawTable]` — regex `<table>(.*?)</table>` (re.S non-greedy).
  - `TableGrid` (rows list-of-list): expand `colspan`, nối header multi-row theo cột, strip cell.
- `etl/numbers.py`:
  - `parse_vn_number(s) -> float|None` — bỏ dot nghìn, `(x)`→âm, `-`/`–`→None, `%` giữ.
  - `detect_unit(grid, page_text) -> (factor, label)` — ưu tiên regex header `(Nghìn|Triệu|Tỷ|Trăm)?\s*VND`, fallback dòng `Đơn vị( tính)?:|ĐVT:|Đơn vị tiền tệ:`.
  - `normalize_label(s) -> str` — NFD bỏ dấu, hạ thường, bỏ ký tự đặc biệt.
  - `parse_period_header(cell) -> (period_key, year)` — `31/12/2018`→year_end, `1/1/2018`→year_start, `2018`→flow_year, `(trình bày lại)`→restated_cur.
- `etl/statements.py` (bản M1): `classify_statement(grid) -> str|None` — keyword sets ASCII-normalized (xử lý OCR: bỏ dấu nháy `'`); bảng thuyết minh → None (wide tier).
- `etl/catalog_builder.py`: mọi bảng → `table_{N}.csv` + `catalog_tables.csv` + `documents.csv`.
- `scripts/run_etl.py`: chạy toàn corpus, **checkpoint theo ticker** (`data/derived/etl_state.json`), log lỗi `etl_errors.tsv`, parallel 4–8 worker.

**DoD (test vàng M1):**
- HPG 2018 CĐKT: header đúng, tổng TÀI SẢN = tổng NGUỒN VỐN.
- VCB: unit lấy từ header `31/12/2022Triệu VND` → factor `1e6`.
- VJC 2018 separate: "Lãi tiền gửi" nằm ở bảng wide Thu nhập tài chính (không có item_code).
- `parse_vn_number("(11.078.921.256)") == -11078921256`; `parse_vn_number("-") is None`.
- Tổng số bảng toàn corpus ≥ 80K.

### M2 — Facts tier (3 BCTC lõi) (2–3 ngày) — *rủi ro cao nhất*
**Task:**
- `etl/statements.py` (nâng cấp):
  - `find_item_code_col(grid) -> int|None` — cột ≥70% giá trị khớp `^\d{1,3}[a-z]?$` (VAS) hoặc `^[IVXLC]+$|^[A-Z]$|^\d$` (bank).
  - `merge_fragments(tables_in_report) -> list[StatementAsset]` — gom table liên tiếp cùng classify + cùng header signature (số cột + cột năm), bỏ header lặp ("mang sang trang sau"), giữ `src_table_ids`.
  - `build_asset(...)`: dòng dữ liệu → `(item_code, item_label, values theo period)`; label rỗng → kế thừa prefix dòng trước.
- `etl/facts_builder.py`: viết `facts/{report_id}_facts.csv` + gộp `facts_all.csv`; unit → VND.
- Cross-sum validator: `validate_facts(asset)` — nếu label chứa công thức (`60 = 50 - 51 - 52`) log warning, **giữ giá trị gốc**.

**DoD (test vàng M2):**
- HPG 2018 LNST mã `60` = `8.600.550.706.227` VND.
- VCB mã chữ `I..XII` parse đúng; bank khó → fallback chỉ giữ mã chính (ghi rõ trong log).
- `facts_all.csv` ≥ 400K dòng; không có report nào 0 facts khi có 3 BCTC lõi.

### M3 — Retrieval (2 ngày)
**Task & chữ ký hàm:**
- `retrieval/entity.py`:
  - `extract_tickers(q) -> list[str]` — regex `\bTICKER\b` case-insensitive + trong ngoặc `(VJC)`; sort theo độ dài giảm dần (tránh `VIC` vs `VICOSTONE`).
  - `company_to_ticker(name) -> str|None` — map tên công ty (diacritic-insensitive) → ticker (cho 364 câu không ticker).
  - `extract_years(q) -> list[int]` — regex `20\d\d`, range `2018–2024`.
  - `extract_report_type(q) -> str|None` — `"công ty mẹ"`→separate, `"hợp nhất"`→consolidated; default truy vấn cả 2.
  - `extract_units(q) -> list[str]`.
- `retrieval/corpus.py`: 3 view — doc/table/facts (text chuẩn hoá diacritic-insensitive).
- `retrieval/bm25.py` (rank_bm25), `dense.py` (BGE-M3 + FAISS), `rrf.py` (k=60), `rerank.py` (bge-reranker-v2-m3 top-100 → top-10).
- `retrieval/facts_index.py`: index `facts_all.parquet`; `get_facts(ticker, years, statement, item) -> DataFrame`.
- `retrieval/pipeline.py`: `search(question, top_k=10) -> SearchResult{report_id, table_id, page_no, unit, snippet}`.
- `scripts/build_retrieval_index.py` chạy offline.

**DoD:** entity filter đúng 100% câu có ticker (dev set); trên 40 câu dev: recall@10 ≥ 0.8 (ước lượng, tune sau ở M7).

### M4 — Sandbox chạy pandas (0.5–1 ngày)
**Task & chữ ký hàm:**
- `sandbox/ast_check.py`: `check_code(code) -> (ok, error)` — AST walk: block `Import/ImportFrom`; block call `{open,eval,exec,compile,input,globals,locals,vars,breakpoint,__import__,getattr,setattr,delattr}`; block attribute `{__class__,...}`; block tên module `{os,sys,subprocess,socket,urllib,ctypes,tempfile,shutil,pathlib,io}`; limit node ≤300, len ≤4000.
- `sandbox/paths.py`: `resolve_evidence_path(csv_path, root) -> Path` — `realpath` nằm trong root + bắt đầu `data/`.
- `sandbox/executor.py`: `run_pandas(code, evidence: dict[var, csv_path], root) -> {ok, result|error, stdout}` — subprocess `python -I runner` (stdin=code, timeout 20s); runner đọc CSV `dtype=str` → `df`/`dfs`, exec với globals hạn chế (whitelist builtins, `pd/np/math/re/json` inject), code phải gán `result`, in JSON kết quả.

**DoD (escape tests):** `import os`, `__import__`, `open()`, `__class__`, `socket`, vòng lặp vô hạn → đều fail/timeout; code hợp lệ trả đúng result; `csv_path` ngoài root bị chặn.

### M5 — Agent (3–4 ngày)
**Task & chữ ký hàm:**
- `agent/llm.py`: `ChatLLM.complete(messages, temperature=0, max_tokens) -> str`; `OpenAICompatibleLLM` (base_url+api_key+model) + `VLLMLLM` (cùng protocol `/v1/chat/completions`); retry 3, timeout 60s.
- `agent/tools.py` (registry + exec):
  - `search_tables(query, ticker=[], years=[], report_type=None, top_k=10)`
  - `get_facts(ticker, years, statement, items=[{code|label}], scope=None)`
  - `inspect_table(report_id, table_id)` → header + sample 5 rows + unit
  - `run_pandas(code)` → chạy sandbox, trả result/error
  - `finalize(answer, docs, tables, evidence, pandas_query)` → `validation.py` re-exec ngay, fail thì quay loop 1 lần
- `agent/react.py`: ReAct loop — history (question, actions, observations, truncate obs ≤2500 + `(truncated)`); parse JSON action `{"tool":..,"args":..}` / `{"final":..}`; `STEP_BUDGET=10`, `temperature=0`; lỗi parse ≤2 liên tiếp → buộc finalize fallback; `run_pandas` crash → feed error + retry ≤2.
- `agent/prompts.py`: system prompt (tool schema + runtime contract: `df`/`dfs`, builtin allowlist, không import, round cuối 2dp, gán `result`; data contract: `dtype=str`, `.` nghìn / `,` thập phân / `( )` âm; reliability: không round trung gian, check filter rỗng trước `.iloc[0]`) + **few-shot 3 mẫu** (tra cứu facts / wide + đổi đơn vị / ratio đa công ty).
- `agent/units.py`: `unit_to_factor(label)`, `convert_value(v, from_unit, to_unit)`.
- `agent/validation.py`: answer numeric finite; unit plausibility (hỏi "triệu đồng" mà result ≈ VND gốc lệch >1e5 → cảnh báo observation).
- `scripts/run_batch.py`: chạy N câu, `checkpoint.jsonl`, concurrency 4–8.

**DoD:** chạy 20 câu ngẫu nhiên: crash rate < 30%; câu tra cứu đơn đúng ≥ 60% (ước lượng); **mọi câu finalize đều re-exec khớp answer 100%**.

### M6 — Submission builder + validate + pack (1–1.5 ngày)
**Task:**
- `submission/builder.py`: gom output agent → record `{id, question, answer, relevant_docs, relevant_tables, evidence, pandas_query}`; materialize evidence CSVs vào `data/out/data/` (dedupe theo hash); **assert đủ 1,012 id**; fallback câu fail → vẫn ghi record (answer = giá trị hợp lệ nhất + pandas_query khớp), không bỏ sót câu.
- `submission/validate.py`: với mỗi record — `executor.run(query, evidence)` trên CSV thật trong `data/out/data/` → so `answer` (abs tol 0.01); kiểm tra `relevant_docs/tables` tham chiếu tồn tại; `csv_path` bắt đầu `data/`; xuất `validation.jsonl` + summary `total/ok/crash/mismatch/no_evidence`.
- `submission/pack.py`: ZIP với `submission.json` + `data/**` **ở root** (không bọc thư mục cha), đúng 1 file `.json`, sort entries, ghi checksum.

**DoD:** đủ 1,012 id; validate re-exec 100% khớp answer; ZIP mở lại được (zipfile) đúng cấu trúc.

### M7 — Dev set + mock-eval + tuning (3–4 ngày, xen kẽ)
**Task:**
- `eval/devset.py`: chọn ~40 câu phủ 7 nhóm (tra cứu lõi / thuyết minh / đa năm / đa công ty / ratio / argmax / đổi đơn vị).
- `scripts/label_dev.py`: agent draft → human verify **2 vòng** → `data/out/devset_labels.json` (gitignored): `{id, gold_answer_float, gold_report_ids, gold_table_ids, notes}`.
- `eval/metrics.py`: **đúng công thức BTC** — macro Retrieval Precision/Recall/F2 (`F2 = 5PR/(4P+R)`), Answer Accuracy (tol 0.01), Execution Accuracy (crash).
- `eval/runner.py`: chạy pipeline trên dev set → báo cáo metrics; chạy lại sau mỗi lần đổi prompt/tool/ETL.

**DoD:** có baseline metrics; phân tích failure mode (crash/wrong unit/sai bảng/multi-company) → điều chỉnh prompt, retrieval k, entity rules, synonym map.

### M8 — Full run + nộp + paper (2 ngày)
**Task:**
- Chạy 1,012 câu: `run_batch.py` (checkpoint, cache retrieval, concurrency), `validate.py`, `pack.py`.
- Nộp thử 1 bài sớm đọc leaderboard (10 bài/ngày, private 5 bài).
- Viết working notes paper: phương pháp, **model provenance** (Qwen3.5-9B: link HF + cách lấy weights + provider API), nguồn dữ liệu (ViFinQA/TiniX, CC BY-NC), quy ước `relevant_tables` (R1).

**DoD:** ZIP hợp lệ đã nộp; paper nộp; ghi chú nguồn đầy đủ.

---

## 5. Rủi ro & chiến lược

- **R1 — Quy ước `relevant_tables` (`|350`)**: max quan sát 248/report ⇒ không phải ordinal per-report. Chiến lược: (1) dùng `report_id|table_N` theo `make_table_ref` của paper; (2) `TablePositionEncoder` pluggable (`per_report_ordinal` default, `page_no`, `line_no`, `corpus_global`) — đổi qua config; (3) mọi `relevant_tables` ánh xạ tới evidence CSV thật ⇒ Answer/Execution Accuracy không bị ảnh hưởng; (4) ghi giả định trong paper; (5) nộp thử sớm đọc leaderboard.
- **R2 — OCR lỗi dấu/ký tự**: `normalize_label` diacritic-insensitive; ưu tiên `item_code`, fallback label synonym; cross-sum validate ghi log để đo chất lượng facts.
- **R3 — Báo cáo thiếu năm được hỏi**: entity filter không hard-fail; trả facts trống + agent xử lý.
- **R4 — Câu hard đa bước vượt step budget**: few-shot dạy "giải facts_all trước, filter sau"; fail → fallback không crash (ưu tiên Execution Accuracy).
- **R5 — API/provider**: cần xác minh provider API có serve **Qwen3.5-9B** (kiến trúc linear-attention mới) + vLLM hỗ trợ inference trước khi thuê GPU; dự phòng Qwen3-8B/14B (4/2025) — đổi 1 dòng config. Ghi nguồn lấy model trong paper.
- **R6 — Tolerance đáp án chưa rõ**: giả định abs 0.01 (paper); mọi câu re-exec kiểm tra answer trên CSV thật.

---

## 6. Verification (cách test end-to-end)

1. **Unit test ETL**: test vàng M1/M2 (HPG LNST 8.600.550.706.227; VJC "Lãi tiền gửi" wide-only; parse `(…)`/`-`/`.`; unit header bank).
2. **Sandbox test**: escape cases chặn hết; timeout hoạt động.
3. **Re-exec validation (bắt buộc trước nộp)**: `validate.py` chạy lại 100% `pandas_query` trong sandbox trên đúng CSV `data/out/data/` → so answer tol 0.01; summary `total/ok/crash/mismatch/no_evidence`.
4. **Dev set mock-eval**: ~40 câu gán nhãn → macro P/R/F2 + Answer + Execution Accuracy (công thức BTC).
5. **Packaging check**: ZIP đúng 1 `submission.json` + `data/**` ở root, đủ 1,012 id, mọi `csv_path` bắt đầu `data/` và tồn tại.
6. **Smoke thật**: M5 chạy 20–50 câu ngẫu nhiên, đo crash rate trước khi scale.

**Critical files**: `etl/statements.py` (trích 3 BCTC lõi + merge + item_code), `etl/numbers.py` (parse số/unit — đúng đơn vị = đúng đáp án), `agent/react.py` (ReAct + self-correction), `sandbox/executor.py` (dùng chung run_pandas + validate), `submission/builder.py` (lắp ráp + ZIP hợp lệ).
