---
name: system-evaluator
description: >-
  Universal Quad-Node Cross-Verification and Ground-Truth Audit Skill.
  Reconciles Documentation (claims, metrics, schemas) vs Real Data/State (databases,
  files) vs Source Code (logic, query patterns) vs Benchmarks/Tests (golden sets).
  Detects data drift, schema desync, query blindspots, and invalid assertions across
  ANY project (RAG, Text-to-SQL, ETL, Web Backend, Data Pipelines).
  Use when the user asks to "audit", "evaluate", "thẩm định hệ thống", "kiểm tra chéo",
  "đánh giá hệ thống", or invokes /system-evaluator.
---

# Universal System Evaluator & Ground-Truth Audit Framework

The `system-evaluator` skill enforces rigorous, evidence-based cross-verification across the four fundamental nodes of any software/AI engineering system:

```
                  [1. Specification & Docs]
                     (Markdown specs, README, API contracts)
                                 ▲
                                / \
                               /   \
                              /     \
    [2. Execution Code] ◄────┼───────┼────► [3. Persistent Data / State]
    (Source code, Logic,      \     /        (DBs, Tables, Schemas,
     Pipelines, Queries)       \   /          Indexes, Assets, Files)
                               \ /
                                ▼
                   [4. Benchmarks & Tests]
                     (Golden test sets, Assertions,
                      Ground-truth datasets)
```

--------------------------------------------------------------------------------

## The 5 Core Invariants (Quy tắc Thẩm định Bất biến)

1. **Rule of Provenance (Mọi khẳng định phải có bằng chứng)**:
   Never accept any number, percentage, schema constraint, or claim in documentation without executing a deterministic verification command against real data/state.
2. **Rule of Contract Parity (Tính nhất quán hợp đồng)**:
   `Doc Schema == Database/Storage Schema == Code Models (Pydantic/DTO) == Test Fixtures`. Any delta is an immediate finding.
3. **Rule of Edge-Case Realism (Thử nghiệm trên dữ liệu thật)**:
   Regex patterns, date filters, wildcard queries, and normalization logic must be tested against actual data distributions in the storage layer to catch blindspots (e.g. unpadded dates, collation/accent discrepancies, null handling).
4. **Rule of Benchmark Precision (Thống kê chính xác, cấm ước lượng)**:
   Test counts, class distributions, and unit frequencies must be computed dynamically by parsing test files or benchmark datasets. Never write subjective estimates.
5. **Rule of Minimalist Remediation (Báo cáo kèm bản vá tối giản)**:
   Every flagged discrepancy must include: File, Line number, Current Doc Claim, Ground-Truth Reality (with command proof), and an exact replacement chunk or git diff.

--------------------------------------------------------------------------------

## 4-Phase Universal Audit Workflow

### Phase 1: Discovery & Claims Extraction
Identify and catalog the system's artifacts:
- **Docs Node**: Scan `docs/`, `specs/`, `README.md` for:
  - Quantitative claims: table row counts, percentage distributions, performance figures.
  - Schema definitions: table names, column lists, types, constraints.
  - Query examples & pattern templates.
  - Sample case studies and provenance citations.
- **Data Node**: Locate primary storage (`*.db`, SQLite, Postgres connection, Parquet, CSV, JSONL).
- **Code Node**: Locate pipelines, query builders, API endpoints, data models.
- **Benchmark Node**: Locate test suites (`tests/`), questions/eval sets (`*.jsonl`, `*.csv`).

### Phase 2: Independent Deterministic Verification
Execute zero-dependency audit scripts or direct CLI/SQL commands to measure the true system state:
- Row counts, column types, distinct values, null counts from persistent storage.
- Keyword frequencies, test case counts from benchmark files.
- Execute query patterns (e.g. `LIKE '%01/01%'` vs `LIKE '%1/1/%'`) on real data to detect missing records.

### Phase 3: Drift & Desync Triangulation
Compute the delta matrix:
- `Data Drift`: Stale numbers in docs following data ingestion or DB rebuilds.
- `Schema Desync`: Columns or types documented that do not match the real database or code models.
- `Query Blindspots`: Hardcoded patterns in docs or code that fail on real data edge-cases.
- `Benchmark Mismatch`: Unit tables or category distributions in docs that deviate from the benchmark.
- `Cross-Doc Contradictions`: Inconsistencies between different specification documents.

### Phase 4: Audit Reporting & Minimalist Diff
Output the findings using the Standard Audit Report format (below) and propose precise, atomic fixes.

--------------------------------------------------------------------------------

## Standard Audit Report Format

When delivering an evaluation report, always follow this structured format:

```markdown
# 📊 BÁO CÁO THẨM ĐỊNH HỆ THỐNG (SYSTEM EVALUATION REPORT)

### 1. Bảng Ma trận Đối chiếu (Ground-Truth Drift Matrix)
| Hạng mục / Khẳng định | Công bố trong Docs | Thực tế kiểm chứng | Lệnh / Query kiểm chứng | Trạng thái |
| :--- | :--- | :--- | :--- | :---: |
| [Mô tả chỉ tiêu/schema] | [Giá trị trong docs] | [Giá trị thực tế] | `[Lệnh thực thi]` | ✅ KHỚP / ❌ LỆCH |

### 2. Chi tiết các Điểm lệch & Rủi ro tiềm ẩn (Identified Findings)
- **Finding 1: [Loại lỗi: Data Drift / Query Blindspot / Contract Desync]**
  - Vị trí: `[File]:[Dòng]`
  - Mô tả: [Chi tiết sai lệch]
  - Nguyên nhân gốc rễ (Root Cause): [Tại sao xảy ra]
  - Bằng chứng thực nghiệm: [Output của lệnh query/kiểm thử]

### 3. Đề xuất Bản vá Tối giản (Actionable Remediation Diff)
[Code/Doc diff sẵn sàng áp dụng]
```
