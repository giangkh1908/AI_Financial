# Kế hoạch triển khai: Agentic RAG cho Text-to-Pandas trên BCTC (ViFinQA)

## Context

Cuộc thi ViFinQA yêu cầu xây hệ thống AI: với mỗi câu hỏi tài chính tiếng Việt (1,012 câu), xác định đúng bảng BCTC (table retrieval) và sinh pandas query thực thi được (text-to-pandas), nộp dưới dạng `submission.json` + thư mục `data/` (chứa CSV evidence) → ZIP. **Grader sẽ chạy lại `pandas_query` trên đúng CSV ta nộp và kiểm tra `answer`** → tính tái lập là ràng buộc cấu trúc, không chỉ mục tiêu.

**Đã khảo sát dữ liệu thật (grounding):**
- `data/code_stock.csv` (100 mã), `data/questions/questions.jsonl` (1,012 câu, không đáp án), `data/financial_statements/{TICKER}/{YEAR}/{report_id}/{report_id}_extracted.txt` (~100 ticker × 2015–2025 × consolidated/separate; OCR có `===== PAGE N =====` + bảng HTML `<table><tr><td>`, lỗi dấu nhiều).
- Bảng: cột đa năm, dòng section `colspan`, cột `Mã số` (VAS: 01–70, 100–500), unit trong dòng `Đơn vị: VND` hoặc ngay trong header (ngân hàng: `31/12/2022Triệu VND`), số dấu chấm nghìn / âm trong ngoặc / `-`.
- **2 profile schema**: công nghiệp (mã số dạng số VAS) vs **ngân hàng** (mã chữ La Mã `I, II, VI`, unit trong header) → facts tier phải hỗ trợ cả 2.
- Bảng bị cắt trang → phải merge fragment cùng schema (KQKD tách 2 table, LCTT tách 3).
- Bảng thuyết minh (notes) KHÔNG có item_code ("Lãi tiền gửi" của VJC nằm trong bảng Thu nhập tài chính) → wide raw tier là bắt buộc.
- Số table/report tối đa quan sát = 248 (OCB 2025) → ví dụ `|350` của BTC **không phải** ordinal per-report (xem Rủi ro R1).
- Câu hỏi: 472 câu 1 ticker, **364 câu không có ticker** (chỉ tên công ty), 165 câu 3+ ticker; ticker viết thường; câu hard đa bước (ROE tại năm CFO/DT cao nhất giữa nhiều công ty); đơn vị triệu/trăm tỷ/nghìn tỷ đồng.

**Quyết định đã chốt với chủ dự án:**
- LLM **Qwen3.5-9B-Instruct** (`Qwen/Qwen3.5-9B`; Apache 2.0, 9B ≤14B, phát hành **2/3/2026 < 1/6/2026** → hợp lệ; hybrid Gated DeltaNet + Gated Attention, context 262K, tool calling); giai đoạn đầu dùng **API** (provider mở, OpenAI-compatible), sau thuê GPU chạy local vLLM cùng model — client provider-agnostic, đổi qua config.
- Agent framework: **custom ReAct** (JSON action text-based), không LangGraph/LlamaIndex.
- **BẮT BUỘC có sandbox chạy pandas_query** (subprocess + hạn chế builtins/import + chặn network/file-write + timeout), dùng chung cho tool `run_pandas` và validator lúc đóng gói.
- Embedding BGE-M3 + reranker bge-reranker-v2-m3 (chạy CPU local).
- 2 tầng evidence: **facts long-format** (3 BCTC lõi, có item_code, chuẩn VND) + **wide raw CSV** (mọi bảng còn lại).

---

## Kiến trúc tổng thể

```
data/financial_statements/**/_extracted.txt  (input read-only)
  │
  ▼  ETL offline (1 lần, checkpoint theo ticker)
parser (PAGE split + table grid) → statements (classify + item_code + merge fragment)
  → facts/{report_id}_facts.csv (long, VND)  +  tables/{report_id}/table_{N}.csv (wide raw)
  → catalog_tables.csv + documents.csv + facts_all.csv (~480K dòng)
  │
  ▼  Retrieval index (offline)
entity (ticker/company/years/report_type/unit) → BM25 + BGE-M3 dense → RRF → rerank → top-k
  │
  ▼  Agent (per question, online) — custom ReAct, Qwen3.5-9B
tools: search_tables | get_facts | inspect_table | run_pandas | finalize
self-correction ≤2 retries; unit conversion; validation feedback
  │
  ▼  Submission (offline)
builder (submission.json + data/*.csv) → validate (re-exec mọi query) → pack ZIP
```

**Nguyên tắc 2 tầng evidence:**
- **Tier A — facts long-format** (`{report_id}_facts.csv`): 3 BCTC lõi, có `item_code`, chuẩn hoá về VND → bền với OCR noise, workhorse cho ratio/đa năm/đa công ty/argmax.
- **Tier B — wide raw** (`tables/{report_id}/table_{N}.csv`): mọi bảng còn lại (notes/thuyết minh), giữ nguyên chuỗi OCR → cho chỉ tiêu không có mã.

---

## Module layout

```
D:\GURU\
├── pyproject.toml / requirements.txt      # pandas, numpy, rank-bm25, sentence-transformers,
│                                          # FlagEmbedding, faiss-cpu, httpx, pydantic, pyyaml
├── configs/
│   ├── base.yaml      # paths, retrieval k=10, sandbox timeout=20s, tolerance=0.01
│   ├── api.yaml       # provider=openai-compatible, base_url, model=Qwen3.5-9B-Instruct
│   └── local_vllm.yaml# provider=vllm, base_url=http://localhost:8000/v1
├── src/vifinqa/
│   ├── config.py  constants.py            # UNIT_FACTORS, STEP_BUDGET=10, k, tol
│   ├── etl/
│   │   ├── parser.py       # split_pages, extract_tables, TableGrid (colspan/rowspan)
│   │   ├── numbers.py      # parse_vn_number, detect_unit, normalize_label (diacritic-insensitive)
│   │   ├── statements.py   # classify_statement (2 profile), find_item_code_col, merge_fragments
│   │   ├── facts_builder.py / catalog_builder.py / run.py
│   ├── retrieval/
│   │   ├── entity.py  corpus.py  bm25.py  dense.py  rrf.py  rerank.py  facts_index.py  pipeline.py
│   ├── agent/
│   │   ├── llm.py          # ChatLLM: OpenAICompatibleLLM | VLLMLLM (chung interface)
│   │   ├── tools.py  react.py  prompts.py  units.py  validation.py
│   ├── sandbox/
│   │   ├── ast_check.py    # AST allowlist/blocklist, limit node/length
│   │   ├── executor.py     # subprocess python -I runner + timeout; đọc-only, resolve paths an toàn
│   │   └── paths.py
│   ├── submission/
│   │   ├── builder.py  validate.py  pack.py
│   └── eval/
│       ├── devset.py  metrics.py  runner.py
├── scripts/  run_etl.py  build_retrieval_index.py  run_batch.py  label_dev.py  validate_submission.py
├── tests/    test_numbers.py  test_parser.py  test_statements.py  test_sandbox.py  test_entity.py  test_submission_roundtrip.py
└── data/derived/ + data/out/     # đều gitignored (nằm trong data/)
```

---

## Schema core

**Facts** `data/derived/facts/{report_id}_facts.csv`:
`ticker, year, report_type, statement, item_code, item_label, item_label_raw, period_key, period_label, value_vnd, src_table_ids`
- `statement ∈ {balance_sheet, income, cash_flow}`; `report_type ∈ {consolidated, separate}`.
- `period_key ∈ {year_start, year_end, flow_year, restated_cur, restated_prev}` (parse từ header: CĐKT `31/12/2018`→year_end, `1/1/2018`→year_start; KQKD/LCTT `2018`→flow_year; bank `(trình bày lại)`→restated_cur).
- `item_code`: VAS số (`110`,`411a`) hoặc bank (`I`,`VI`,`A`); `""` cho dòng tổng không mã (vẫn giữ, có nhãn).
- `value_vnd`: float chuẩn về VND (unit từ header/dòng `Đơn vị:`); `src_table_ids` → dùng cho `relevant_tables`.

**Wide raw** `data/derived/tables/{report_id}/table_{N}.csv`: grid giữ nguyên chuỗi OCR (header multi-row nối dọc), KHÔNG parse số. `N` = số thứ tự `<table>` trong report (1-based).

**Catalog** `data/derived/catalog_tables.csv`: `report_id, ticker, year, report_type, table_id, page_no, unit, is_statement, statement, header_text, row_labels, n_rows, n_cols, anchor_context` (anchor ≈ 6 dòng text trước bảng — chứa tên mục/đơn vị).

**Evidence per-question** (trong ZIP `data/`):
- `data/facts__{qid}__{subset_key}.csv` — subset facts_all cho đúng ticker/stmt/năm (vài trăm dòng).
- `data/table__{report_id}__{N}.csv` — bảng wide dùng.
- `evidence = [{"variable":"df1","csv_path":"data/..."}, ...]`; 1 bảng → biến `df`, nhiều → `dfs['df1']`.

**submission.json**: đủ 1,012 record `{id, question, answer(float), relevant_docs[], relevant_tables[], evidence[], pandas_query}`. `pandas_query`: chỉ dùng `pd/np/math/re/json` + builtin allowlist, **không import**, gán `result` (theo runtime contract của paper).

---

## Milestone & nỗ lực

| # | Nội dung | Nỗ lực | Rủi ro chính / giảm thiểu |
|---|---|---|---|
| **M0** | Scaffolding: pyproject, config, constants, loader | 0.5 ngày | Thấp |
| **M1** | ETL parser + numbers + wide tables + catalog | 2–3 ngày | Header đa dòng bank/unit trong header → unit test 3 profile (HPG, VCB, VJC) |
| **M2** | **Facts tier** (3 BCTC lõi): classify, merge fragment, item_code, validate cross-sum | 2–3 ngày | **Rủi ro cao nhất**: continuation/trùng mã/OCR tách dòng → test vàng HPG+VCB; bank khó → fallback chỉ lấy mã chữ chính |
| **M3** | Retrieval: entity + BM25 + BGE-M3 + RRF + rerank + facts_index | 2 ngày | 364 câu không ticker → map tên công ty diacritic-insensitive |
| **M4** | **Sandbox**: ast_check + executor + paths + tests | 0.5–1 ngày | Bảo mật → test escape trên Windows |
| **M5** | Agent loop: tools + react + prompts + units + self-correction | 3–4 ngày | Model 14B sai tool JSON/đơn vị → few-shot + validation feedback |
| **M6** | Submission builder + validate + pack | 1–1.5 ngày | **Thiếu câu = mất cả bài** → assert đủ 1,012 id |
| **M7** | Dev set (~40 câu tự gán nhãn) + mock-eval + tune | 3–4 ngày (xen kẽ) | Gold tự gán sai → human verify 2 vòng |
| **M8** | Full run 1,012 câu + validate + nộp thử + paper | 2 ngày | API cost/latency → checkpoint + concurrency + cache retrieval |

**Tổng ~15–20 ngày làm việc.** Ưu tiên: M1–M2 (ETL) trước — mọi thứ phụ thuộc tầng dữ liệu sạch; test vàng từ đầu để bắt OCR lỗi sớm.

**Critical files**: `etl/statements.py` (trích 3 BCTC lõi + merge + item_code), `etl/numbers.py` (parse số/unit — đúng đơn vị = đúng đáp án), `agent/react.py` (ReAct + self-correction), `sandbox/executor.py` (dùng chung cho run_pandas và validate), `submission/builder.py` (lắp ráp + ZIP hợp lệ).

---

## Rủi ro & chiến lược

- **R1 — Quy ước `relevant_tables` (`|350`)**: max quan sát 248/report ⇒ không phải ordinal per-report. Chiến lược đa lớp: (1) dùng `report_id|table_N` theo convention `make_table_ref` của paper (BTC xây test trên cùng pipeline paper); (2) `TablePositionEncoder` pluggable (`per_report_ordinal` default, `page_no`, `line_no`, `corpus_global`) — đổi qua config nếu BTC công bố khác; (3) mọi `relevant_tables` luôn ánh xạ tới evidence CSV thật ⇒ **Answer + Execution Accuracy không bị ảnh hưởng**; (4) ghi giả định trong paper; (5) nộp thử 1 bài sớm đọc leaderboard để đoán convention.
- **R2 — OCR lỗi dấu/ký tự**: `normalize_label` diacritic-insensitive; matching ưu tiên `item_code`, fallback label synonym; cross-sum validate ghi log để đo chất lượng facts trước khi nộp.
- **R3 — Báo cáo thiếu năm được hỏi** (ticker chỉ có từ năm X): entity filter không hard-fail; trả facts trống + agent xử lý.
- **R4 — Câu hard đa bước** vượt step budget: few-shot dạy "giải facts_all trước, filter sau"; nếu vẫn fail → fallback không crash (ưu tiên không để Execution crash).
- **R5 — API**: chọn provider phục vụ model mở (luật chỉ cấm model đóng); ghi nguồn lấy model trong paper; sau thuê GPU chạy local vLLM cùng model — chỉ đổi config. ⚠️ Qwen3.5-9B là kiến trúc hybrid linear-attention (Gated DeltaNet) — cần **xác minh** provider API có serve model này và **vLLM có hỗ trợ inference** trước khi thuê GPU; nếu chưa có, phương án dự phòng là Qwen3-8B/14B (phát hành 4/2025, hỗ trợ rộng rãi).
- **R6 — Tolerance đáp án chưa rõ**: giả định abs 0.01 (paper); mọi câu đều re-exec kiểm tra `answer` trên CSV thật.

---

## Verification (cách test end-to-end)

1. **Unit test ETL**: test vàng — HPG 2018 LNST (mã 60) = 8.600.550.706.227 VND; VJC 2018 "Lãi tiền gửi" chỉ nằm ở wide tier; parse số `(11.078.921.256)`/`-`/`.` đúng; unit trong header bank (VCB).
2. **Sandbox test**: escape cases (import os/socket, open file, eval, attribute dunder) đều bị chặn; timeout hoạt động.
3. **Re-exec validation (bắt buộc trước nộp)**: `validate.py` chạy lại 100% `pandas_query` trong sandbox trên đúng CSV `data/out/data/` → so `answer` tol 0.01; summary `total/ok/crash/mismatch/no_evidence`.
4. **Dev set mock-eval**: ~40 câu gán nhãn thủ công → tính macro Retrieval P/R/F2 + Answer Accuracy + Execution Accuracy (đúng công thức BTC); chạy lại sau mỗi lần đổi prompt/tool/ETL.
5. **Packaging check**: `pack.py` + kiểm tra ZIP có đúng 1 `submission.json` + `data/**` ở root, đủ 1,012 id, mọi `csv_path` bắt đầu `data/` và tồn tại; thử mở lại bằng `zipfile`/đọc lại `submission.json`.
6. **Smoke chạy thật**: M5 chạy 20–50 câu ngẫu nhiên, đo crash rate trước khi scale.
