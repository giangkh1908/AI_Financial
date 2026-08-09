# ARCHITECTURE — Kiến trúc & Luồng xử lý của GURU (ViFinQA)

Tài liệu này mô tả **toàn bộ hệ thống** theo 3 luồng chính:
1. **ETL** — biến OCR báo cáo tài chính thành dữ liệu sạch (offline, chạy 1 lần khi có dữ liệu mới).
2. **Retrieval** — tìm đúng bảng chứa câu trả lời cho 1 câu hỏi.
3. **Answering** — sinh `pandas_query` chạy trên bảng đó, trả đáp án số.

Kèm theo: kiến trúc dữ liệu (7 tầng), config, thư mục, và các quyết định thiết kế quan trọng.

---

## 0. Bức tranh 1 nét

```
┌─────────────────────────────────────────────────────────────────────┐
│  OFFLINE (build 1 lần / khi có dữ liệu mới)                          │
│                                                                     │
│  OCR BCTC ──► ETL ──► Bảng sạch ──► Embed ──► Qdrant (vector DB)     │
│  (txt thô)            (canonical)      (nemotron)                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ONLINE (mỗi câu hỏi)                                               │
│                                                                     │
│  Câu hỏi ──► Entity ──► Embed ──► Qdrant search ──► top-k bảng      │
│                │                              │                     │
│                ▼                              ▼                     │
│         Deterministic                     Codegen (LLM)             │
│         (match label,                       (sinh pandas_query)     │
│          không cần LLM)                          │                  │
│                │                                ▼                   │
│                ▼                           Sandbox (chạy thử)       │
│          Đáp án ◄───────────────────────────────┘                   │
│                │                                                    │
│                ▼                                                    │
│         Builder → submission.json + data/*.csv                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Ý tưởng cốt lõi:** *"Đừng bắt LLM đọc cả báo cáo. Hãy tìm đúng bảng chứa số liệu trước (retrieval), rồi mới cho LLM viết query ngắn chạy trên bảng đó."*

---

## 1. Luồng ETL — OCR → Dữ liệu sạch (offline)

### 1.1. Mục tiêu
Biến 1,965 file OCR HTML thô thành **bảng chuẩn hoá** mà bất kỳ tầng nào (facts/evidence/index/LLM) cũng đọc được — không ai tự đoán cột lại.

### 1.2. Các bước

```
Mỗi file {rid}_extracted.txt
   │
   ▼
split_pages: tách trang theo "===== PAGE N ====="
   │
   ▼
parse_table_grid: mỗi <table> → grid 2D, expand colspan+rowspan, decode entity
   │
   ▼
format_classify: detect TableLayout (cột nào là mã/label/kỳ/đơn vị)
   │
   ▼
catalog_builder: ghi 3 thứ
   ├─► tables/{rid}/table_N.csv        ← WIDE (grid thô, chưa parse số)
   ├─► catalog_tables.csv              ← metadata mọi bảng (23 cột)
   └─► layouts/{rid}.json              ← layout chi tiết từng bảng
   │
   ▼
tidy: wide + layout → evidence/{rid}__table_N.csv   [chi_tieu, Mãsố, ky, value]
facts: 3 BCTC lõi → facts_all.csv                    (long-format VND)
merged: facts → evidence_merged + statement_meta     (statement gộp)
```

**Điểm mấu chốt:** wide CSV là **canonical** — mọi tầng khác derive từ nó qua `TableLayout`. Không ai đọc lại OCR.

### 1.3. Các script

| Script | Tạo ra | Thời gian |
|---|---|---|
| `run_etl.py` | wide + catalog + layouts | ~12p |
| `rebuild_evidence.py` | evidence (tidy mọi bảng) | ~13p |
| `run_facts.py` | facts_all.csv | ~15p |
| `run_merged_evidence.py` | evidence_merged + statement_meta | ~2p |
| `build_retrieval_index.py` | embeddings + Qdrant | ~70p (free) |

---

## 2. Luồng Retrieval — Câu hỏi → top-k bảng

```
Câu hỏi
   │
   ▼
extract_entities: ticker, year, report_type, unit, statement-hint
   │        (VJC, 2018, separate, triệu, None)
   ▼
embed query: câu hỏi → vector 2048-dim (nemotron)
   │
   ▼
Qdrant hybrid search:
   │   filter = ticker IN [...] AND year IN [...] AND report_type
   │   dense cosine (HNSW INT8)
   │   + sparse TF (đang tắt)
   │   → statement bonus mềm
   ▼
top-k bảng (k=10) → SearchResult[]
   │
   ▼
_plan_evidence:
   │   bảng statement → evidence_merged (gộp toàn bộ chỉ tiêu)
   │   bảng notes      → evidence tidy per-table
   ▼
usable + cards (đưa vào LLM) + evidence_paths (chạy query)
```

**Chi tiết quan trọng:**
- **Entity** = thông tin để **lọc** (không phải để LLM đoán): ticker/year/report_type giúp thu hẹp 118k bảng về ~một report.
- **k=10**: lấy 10 bảng top (trước là 5 — tăng vì bảng đúng hay rank 6-8).
- **Fallback report_type**: nếu filter quá hẹp (báo cáo `other` không tách cons/sep) → bỏ filter type tìm lại.
- **`use_sparse=False`**: sparse TF gây nhiễu tiếng Việt → chỉ dùng dense.

---

## 3. Luồng Answering — Bảng → Đáp án

```
top-k bảng + evidence_paths
   │
   ├─► Deterministic engine (engine/deterministic.py)   ← ƯU TIÊN, không tốn LLM
   │     lookup đơn giản: 1 ticker + 1 chỉ tiêu
   │     match label trong facts/evidence (token-overlap)
   │     → sinh pandas_query template
   │
   └─► (nếu deterministic không match / câu phức)
         Codegen (LLM): build_messages(cards) → pandas_query
              │
              ▼
         Sandbox (AST check + exec cô lập)
              │  lỗi? retry ≤2 lần, feed error vào LLM
              ▼
         answer
   │
   ▼
Builder: materialize evidence CSV flat + rewrite relevant_tables
         → submission.json
```

**Thiết kế 2 tầng:**
1. **Deterministic trước** — câu lookup đơn giản (khoảng 1/3 câu) không cần LLM: nhanh, chính xác, rẻ.
2. **LLM chỉ cho câu phức** (so sánh, tỷ lệ, argmax, notes không chuẩn) — sinh pandas_query rồi chạy trong sandbox.

**Contract quan trọng:** `pandas_query` chạy trên CSV schema `[chi_tieu, Mãsố, ky, value]`, dùng `df1, df2...` (biến từ evidence), **cấm `dfs["..."]`**, có `vn_num()` nhúng sẵn (grader không inject helper).

---

## 4. Kiến trúc dữ liệu — 7 tầng

```
T0 OCR gốc     {rid}_extracted.txt                     (trang + <table> HTML)
T1 WIDE        tables/{rid}/table_N.csv                (grid rowspan-aligned, số thô)
T2 METADATA    catalog_tables.csv + layouts/{rid}.json (23 cột + layout JSON)
T3 TIDY        evidence/{rid}__table_N.csv             [chi_tieu, Mãsố, ky, value]
T4 FACTS       facts_all.csv                           (3 BCTC lõi, long-format VND)
T5 MERGED      evidence_merged + statement_meta.csv    (statement gộp)
T6 EMBED       embeddings/{ticker}.npy + .json         (cache vector 2048-dim)
   + INDEX     Qdrant collection "bctc_tables"         (118,728 points)
```

| Tầng | Đọc bởi | Dùng để làm gì |
|---|---|---|
| T0 | ETL | nguồn duy nhất |
| T1 | tidy, facts | canonical, không sửa |
| T2 | index, loop, tidy | chunk text, layout, unit, relevant_tables |
| T3 | LLM codegen, sandbox, builder | pandas_query chạy trên đây |
| T4 | deterministic, index (fact hints) | match nhanh + enrichment chunk |
| T5 | loop (plan evidence) | bảng statement gộp toàn bộ |
| T6 | build index | resume không embed lại |

---

## 5. Module chính & trách nhiệm

```
src/vifinqa/
├── etl/                        # OFFLINE: OCR → bảng sạch
│   ├── parser.py               #   tách trang + parse <table> → grid (rowspan)
│   ├── format_classify.py      #   detect TableLayout (code/label/period/unit)
│   ├── catalog_builder.py      #   ghi wide + catalog + layouts
│   ├── tidy.py                 #   wide → evidence 4 cột [chi_tieu, Mãsố, ky, value]
│   ├── facts_builder.py        #   3 BCTC lõi → facts long-format
│   └── merged_evidence.py      #   facts → statement gộp + statement_meta
│
├── retrieval/                  # ONLINE: câu hỏi → top-k bảng
│   ├── entity.py               #   trích ticker/year/report_type/unit
│   ├── index.py                #   chunk text + embed (nemotron) + upsert Qdrant
│   ├── search.py               #   hybrid dense+sparse → SearchResult
│   ├── pipeline.py             #   orchestration: entity → search → bonus → top-k
│   └── facts_index.py          #   truy vấn facts_all (verify + match)
│
├── agent/loop.py               # ONLINE: retrieve → plan evidence → answer
├── engine/deterministic.py     #   lookup không cần LLM (ưu tiên)
├── codegen/                    #   prompt + LLM sinh pandas_query
├── sandbox/                    #   AST check + exec cô lập
└── submission/                 #   builder → submission.json + data/*.csv
```

---

## 6. Config (2 file, merge sâu)

```
configs/base.yaml   ← mặc định chung
configs/api.yaml    ← override cho môi trường API (LLM + embed)
```

| Key | Giá trị | Ý nghĩa |
|---|---|---|
| `retrieval.k` | 10 | top-k bảng trả evidence |
| `retrieval.use_dense/use_sparse` | true/false | dense-only |
| `retrieval.min_n_rows` | 5 | bỏ bảng junk/TOC khỏi index |
| `retrieval.embedding.model` | `nvidia/nemotron-3-embed-1b:free` | model embed (free, dim 2048) |
| `retrieval.embedding.max_chars` | 2000 | cắt chunk text |
| `retrieval.embedding.batch_size/workers` | 100/12 | tốc độ embed |
| `llm.model_id` | `qwen/qwen3.5-9b` | LLM codegen |
| `sandbox.timeout` | 20 | giây chạy pandas_query |

---

## 7. Quyết định thiết kế quan trọng

| Quyết định | Lý do |
|---|---|
| **Wide CSV làm canonical** | 1 nguồn sự thật, cấm re-heuristic → fix bug lệch cột |
| **`TableLayout` từ format_classify** | mọi format (DN/bank/chứng khoán/tiếng Anh) map về 1 layout |
| **Bảng làm đơn vị chunk** | 1 bảng = 1 vector → dễ truy hồi + relevant_tables chuẩn |
| **Schema 4 cột `[chi_tieu, Mãsố, ky, value]`** | khớp grader BTC, query chạy ổn định |
| **Deterministic trước LLM** | ~1/3 câu không cần LLM: nhanh, rẻ, chính xác |
| **nemotron-3-embed-1b:free** | free + tiếng Việt tốt hơn bge-m3 (verify) |
| **dense-only (`use_sparse=False`)** | sparse TF gây nhiễu tiếng Việt |
| **k=10** | bảng đúng hay rank 6-8; đánh đổi input LLM chấp nhận được |

---

## 8. Luồng dữ liệu đầy đủ (mọi file)

```
data/
├── questions/questions.jsonl        1012 câu hỏi (đầu vào)
├── code_stock.csv                   100 mã CK (entity)
├── financial_statements/            OCR gốc 1965 report
└── derived/                          ← toàn bộ sản phẩm ETL
    ├── tables/                       T1 wide (146,246)
    ├── catalog_tables.csv            T2 metadata (146,246 dòng, 23 cột)
    ├── layouts/                      T2 layout JSON (1,973)
    ├── evidence/                     T3 tidy (146,246)
    ├── facts_all.csv + facts/        T4 facts (400,510 dòng)
    ├── evidence_merged/              T5 statement gộp (5,474)
    ├── statement_meta.csv            T5 fragment map (5,474)
    ├── embeddings/                   T6 cache vector (200)
    └── *.json / *.csv               checkpoint + documents.csv
data/out/
    ├── results_*.jsonl               kết quả codegen từng lần chạy
    └── submission_*/                 submission.json + data/*.csv + zip
```
