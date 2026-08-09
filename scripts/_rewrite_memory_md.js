export const meta = {
  name: 'rewrite-memory-md',
  description: 'Verify current MEMORY.md against post-fix codebase + official BTC spec, synthesize corrected full MEMORY.md',
  phases: [
    { title: 'Verify', detail: 'parallel verifiers per slice check current memory vs current code' },
    { title: 'Synthesize', detail: 'rewrite full MEMORY.md with all corrections' },
  ],
}

const ROOT = 'D:\\GURU'
const MEM = ROOT + '\\MEMORY.md'

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    verifier: { type: 'string' },
    discrepancies: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          location: { type: 'string' },
          claim: { type: 'string' },
          actual: { type: 'string' },
          fix: { type: 'string' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
        required: ['location', 'claim', 'actual', 'fix', 'confidence'],
      },
    },
    missing: { type: 'string' },
  },
  required: ['verifier', 'discrepancies', 'missing'],
}

const VERIFIERS = [
  {
    key: 'etl',
    prompt: 'Verify ETL claims in ' + MEM + ' against CURRENT code (post-fix).\n' +
      'Read ' + MEM + ' then check against actual files (read them):\n' +
      '- ' + ROOT + '\\src\\vifinqa\\etl\\numbers.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\etl\\parser.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\etl\\statements.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\etl\\catalog_builder.py   (NEW: has start_line field, table_start_lines() helper, CATALOG_HEADER now 16 cols incl start_line)\n' +
      '- ' + ROOT + '\\src\\vifinqa\\etl\\facts_builder.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\etl\\tidy.py\n' +
      'Check every concrete claim: function signatures, regex names, constants, behaviors. ESPECIALLY verify catalog_builder now has: CatalogRow.start_line field, table_start_lines(full_text) returns {table_idx: physical_line}, CATALOG_HEADER includes "start_line" as 16th column, process_report computes start_lines and passes start_line=start_lines.get(table_idx,0). Also verify catalog_tables.csv on disk now has start_line column (Bash: head -1 ' + ROOT + '\\data\\derived\\catalog_tables.csv) and count rows (wc -l). Verify AAA_financial_statements_2015_consolidated table_2 start_line=214. Return ONLY structured object; list only DISCREPANCIES (claims in memory that are wrong/stale/missing). Put missing codebase facts in "missing".',
  },
  {
    key: 'retrieval',
    prompt: 'Verify retrieval claims in ' + MEM + ' against CURRENT code.\n' +
      'Read ' + MEM + ' then check:\n' +
      '- ' + ROOT + '\\src\\vifinqa\\retrieval\\entity.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\retrieval\\index.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\retrieval\\search.py   (relevant_tables_key STILL returns report_id|table_N — internal, builder converts to line; verify this is still the case)\n' +
      '- ' + ROOT + '\\src\\vifinqa\\retrieval\\facts_index.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\retrieval\\pipeline.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\retrieval\\rerank.py\n' +
      'Check function/class names, Entities fields, extract_tickers 4-stage, FusionQuery RRF, point_id uuid5, _SPARSE_BITS=21, statement_bonus, embed_statement_only=false, rerank disabled, facts_for_table exact match, facts_all.csv no report_id column. Try Bash to count Qdrant points: python -c "from qdrant_client import QdrantClient; c=QdrantClient(url=\\\"http://localhost:6333\\\"); print(c.count(\\\"bctc_tables\\\"))" (may fail if Docker off — note if so). Return ONLY structured discrepancies.',
  },
  {
    key: 'codegen-sandbox',
    prompt: 'Verify codegen + sandbox claims in ' + MEM + ' against CURRENT code.\n' +
      'Read ' + MEM + ' then check:\n' +
      '- ' + ROOT + '\\src\\vifinqa\\codegen\\prompt.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\codegen\\llm.py   (post-fix: generate_query max_retries=2, backoff 2.0s, OpenAI(max_retries=1) hardcoded; thinking=False; _extract_code strips think/fence/imports/def vn_num/dfN reassign)\n' +
      '- ' + ROOT + '\\src\\vifinqa\\sandbox\\ast_check.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\sandbox\\runner.py   (dfs dict keyed by table_ref, df alias when 1 CSV, dtype=str/keep_default_na=False/index_col=None, vn_num inline, _SAFE_BUILTINS)\n' +
      '- ' + ROOT + '\\src\\vifinqa\\sandbox\\executor.py\n' +
      'Check the EXACT sandbox contract and retry/timeout settings (llm.timeout=30, retries=1 in api.yaml). Check ast_check default max_code_len/max_ast_nodes actual values. Check prompt teaches dfs["df1"] or dfs["<table_ref>"] (note which). Return ONLY structured discrepancies.',
  },
  {
    key: 'agent-submission',
    prompt: 'Verify agent + submission claims in ' + MEM + ' against CURRENT code (post-fix).\n' +
      'Read ' + MEM + ' then check:\n' +
      '- ' + ROOT + '\\src\\vifinqa\\agent\\loop.py   (solve max_retries=1 post-fix; _make_record relevant_tables = r.relevant_tables_key() = report_id|table_N internally; _build_table_card; _ensure_tidy)\n' +
      '- ' + ROOT + '\\src\\vifinqa\\submission\\builder.py   (NEW: _load_start_lines, _table_ref_to_line converts rid|table_N -> rid|<start_line>, build() rewrites relevant_tables to line format, returns relevant_tables_rewritten; _VN_NUM_DEF; tidy regenerate fallback)\n' +
      '- ' + ROOT + '\\src\\vifinqa\\submission\\validate.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\submission\\pack.py\n' +
      'Check solve() flow, _make_record evidence csv_path=data/{rid}__{tid}.csv, builder relevant_tables rewrite to line format, validate tol 0.01 short-circuit 0.0 ThreadPool 8, pack ZIP root 1 json. Return ONLY structured discrepancies.',
  },
  {
    key: 'config-scripts-tests',
    prompt: 'Verify config + scripts + tests claims in ' + MEM + ' against CURRENT code (post-fix).\n' +
      'Read ' + MEM + ' then check:\n' +
      '- ' + ROOT + '\\src\\vifinqa\\config.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\loader.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\constants.py\n' +
      '- ' + ROOT + '\\configs\\base.yaml\n' +
      '- ' + ROOT + '\\configs\\api.yaml   (llm.timeout=30.0, retries=1)\n' +
      '- ' + ROOT + '\\scripts\\run_codegen.py   (WALL_CAP = cfg.sandbox.timeout*6.0=120, daemon thread + queue.get timeout)\n' +
      '- ' + ROOT + '\\scripts\\build_submission.py\n' +
      '- ' + ROOT + '\\scripts\\run_etl.py\n' +
      '- ' + ROOT + '\\scripts\\run_facts.py\n' +
      '- ' + ROOT + '\\scripts\\run_retrieval.py\n' +
      'Glob ' + ROOT + '\\scripts\\*.py (should include NEW backfill_start_line.py and smoke_fmt_100.py) and ' + ROOT + '\\tests\\*.py.\n' +
      'Verify all config knob values, WALL_CAP, deep-merge, _load_done checkpoint. Return ONLY structured discrepancies. List NEW scripts not in memory under "missing".',
  },
  {
    key: 'data-state',
    prompt: 'Verify data-state claims in ' + MEM + ' against CURRENT disk (post-backfill).\n' +
      'Read ' + MEM + ' then check disk:\n' +
      '- Bash: ls -la ' + ROOT + '\\data\\derived\\ and ' + ROOT + '\\data\\out\\\n' +
      '- Bash: wc -l ' + ROOT + '\\data\\derived\\catalog_tables.csv ' + ROOT + '\\data\\derived\\facts_all.csv ' + ROOT + '\\data\\derived\\documents.csv\n' +
      '- Bash: head -1 ' + ROOT + '\\data\\derived\\catalog_tables.csv  (verify start_line is now 16th column)\n' +
      '- Read ' + ROOT + '\\data\\derived\\etl_state.json, retrieval_state.json, facts_state.json (whichever exist)\n' +
      '- Read ' + ROOT + '\\data\\out\\codegen_full.log fully\n' +
      '- Bash: ls -la ' + ROOT + '\\data\\out\\submission\\ and count files in submission\\data (ls ' + ROOT + '\\data\\out\\submission\\data | wc -l)\n' +
      '- Read ' + ROOT + '\\data\\out\\results.jsonl first 3 lines, and compute stats: total records, answer zero vs nonzero count, _ok true/false count (python)\n' +
      '- Read ' + ROOT + '\\data\\out\\_fts.txt and _fts2.txt\n' +
      'Verify: catalog 146,246 rows now with start_line column, facts_all 377,578 rows, results.jsonl 1,012 records, submission state. Return ONLY structured discrepancies.',
  },
]

phase('Verify')
const results = await parallel(
  VERIFIERS.map((v) => () => agent(v.prompt, { label: 'verify:' + v.key, phase: 'Verify', schema: SCHEMA }))
)
const discrepancies = VERIFIERS.map((v, i) => ({ verifier: v.key, result: results[i] })).filter((x) => x.result)
log('Verify done: ' + discrepancies.map((d) => d.verifier + '(' + (d.result.discrepancies?.length || 0) + ')').join(', '))

const OFFICIAL_SPEC = [
  'OFFICIAL BTC SUBMISSION SPEC (authoritative, from user 6/8/2026):',
  '- File: single .json, array of records. ZIP = submission.json + data/ at root (no parent folder), exactly 1 .json.',
  '- Record: id (integer), question (string), answer (float), relevant_docs [report_id], relevant_tables ["report_id|<position>"], evidence [{variable, csv_path}], pandas_query (string, re-executable).',
  '- relevant_docs: report_id = final filename in path minus .txt. Example path ocr_filter\\AAA\\2015\\AAA_financial_statements_2015_consolidated -> id AAA_financial_statements_2015_consolidated.',
  '- relevant_tables: "<id_bao_cao>|<vi tri bang trong bao cao>". vi tri = "Vi tri dong bat dau cua bang trong file bao cao OCR tuong ung do BTC cung cap". Example: AAA_financial_statements_2015_consolidated|350. => FORMAT IS report_id|<physical_line_number_of_table_in_OCR>, NOT report_id|table_N.',
  '- evidence.variable: Python-valid DataFrame var name, unique within a question. csv_path: relative, must start with data/, file must exist in zip data/.',
  '- pandas_query: re-executable on standardized data. (Spec example has a stray trailing comma after the query string — a typo, ignore.)',
  '- Missing files or missing questions => submission NOT evaluated AND NOT counted toward daily limit.',
  '- Evaluation (macro-average): Retrieval (Precision, Recall, F2=(5*P*R)/(4*P+R), F2 weights recall more), Answer Accuracy (tol by BTC, paper uses abs 0.01), Execution Accuracy (code runs AND correct / total).',
  '',
  'CRITICAL CORRECTION vs old memory/CLAUDE.md:',
  '- Old memory/CLAUDE.md claimed relevant_tables = report_id|table_N ("da xac minh khop benchmark DSKT-NOWJ/ViFinQA make_table_ref") and that the spec example "|350" was a handwritten error. THIS IS WRONG per the official spec: |350 is a LINE NUMBER, the format is report_id|<line>. The 6/8/2026 "fix" that changed report_id|{position} -> report_id|table_{position} was in the WRONG direction.',
  '- The companion repo DSKT-NOWJ/ViFinQA uses table_N internally (from ocr_filter/*_extracted_tables/table_N.csv) but the OFFICIAL BTC leaderboard grader uses the LINE format per the spec. First submission got TABLES_F2=0.0 with table_N format.',
  '- csv_path filename can still use table_N (spec example: data/AAA_financial_statements_2015_consolidated_table_1.csv) — the filename is our choice; only relevant_tables VALUE must be the line format.',
].join('\n')

const CHANGES_TODAY = [
  'CHANGES MADE THIS SESSION (6/8/2026, after the codegen-retry fixes):',
  '',
  'A. ETL start_line (fix for relevant_tables line-format):',
  '- src/vifinqa/etl/catalog_builder.py: added `start_line: int` field to CatalogRow; added "start_line" as 16th column in CATALOG_HEADER; as_row() appends str(start_line); added helper table_start_lines(full_text) -> {table_idx(1-based): physical_line} using TABLE_RE.finditer + full_text.count(chr(10),0,m.start())+1; process_report computes start_lines once and passes start_line=start_lines.get(table_idx,0) to each CatalogRow. Import TABLE_RE from parser.',
  '- Verified line-counting matches grep -n exactly: AAA_financial_statements_2015_consolidated table_1->line 19, table_2->214 (BS assets), table_3->239, table_4->286 (income), table_5->333 (cash_flow), table_6->433. Each <table> is on one physical line in the OCR txt.',
  '',
  'B. Backfill script (avoid full ETL re-run):',
  '- scripts/backfill_start_line.py (NEW): reads OCR of all 1973 reports, computes table_idx->line, joins into existing catalog_tables.csv adding start_line column. Does NOT re-write 146K wide CSVs. Idempotent.',
  '- RAN: 146,246/146,246 rows got start_line != 0, 0 reports missing OCR. catalog_tables.csv now has 16 columns (start_line is last).',
  '',
  'C. Builder rewrite relevant_tables to line format:',
  '- src/vifinqa/submission/builder.py: added _load_start_lines(derived_dir) -> {(report_id, table_id): start_line}; added _table_ref_to_line(key, start_lines) converting "report_id|table_N" -> "report_id|<start_line>" (falls back to original if start_line missing); build() loads start_lines and rewrites each record relevant_tables to line format; returns new "relevant_tables_rewritten" count. csv_path unchanged (still data/{rid}__{tid}.csv).',
  '',
  'D. Smoke test script (not yet run):',
  '- scripts/smoke_fmt_100.py (NEW): builds 100-question subset submission (prefer answer!=0), validate, pack, then prints FORMAT CHECK + ZIP CHECK against spec. User interrupted before running — NEXT STEP.',
  '',
  'NET EFFECT: relevant_tables output is now report_id|<line> (spec format). csv_path filename still table_N. No Qdrant re-index needed (search stays table_id internally). No codegen re-run needed (results.jsonl keeps table_N; builder rewrites at packaging). To test TABLES_F2: rebuild submission + submit.',
  '',
  'RISK: spec example |350 does NOT match a real <table> line in the AAA 2015 file (table_1 is at line 19, not 350) => the example may be purely illustrative OR BTC counts lines differently. No gold available to verify our line counting matches BTC exactly. But line-format is clearly the literal spec and more correct than table_N (which gave F2=0). If still 0 after resubmit, investigate alternative line-counting schemes ourselves (see Section 6 item A).',
].join('\n')

const SESSION_FINDINGS = [
  'PRIOR SESSION FINDINGS (codegen speed/reliability, earlier 6/8/2026) — keep in MEMORY.md section 7:',
  '',
  'INCIDENT 1: Two duplicate python processes (PID 2080 active, PID 21128 stuck zombie). results_tidy.jsonl had 510 unique IDs, 0 duplicates (zombie never wrote). Killed both.',
  '',
  'INCIDENT 2: Codegen run degraded over time. First run (workers=8, timeout=60s, retries=3, manual max_retries=4, backoff 2^attempt): 550/1012 in ~66min, p95 climbed 47s->210s, fail rate 11%->36%.',
  'Root cause: DOUBLE RETRY LAYER (SDK max_retries=3 + manual max_retries=4 with 2^attempt backoff). One generate_query worst case ~4x60+15=255s; solve up to 3 LLM calls.',
  '',
  '5 FIXES APPLIED (in working tree, codegen re-run not completed to closure):',
  '1. codegen/llm.py: generate_query max_retries 4->2; backoff 2^attempt -> fixed 2.0s+jitter.',
  '2. codegen/llm.py: OpenAI(max_retries=1) hardcoded (was llm.retries=3).',
  '3. configs/api.yaml: llm.timeout 60->30; retries 3->1.',
  '4. agent/loop.py: solve max_retries default 2->1.',
  '5. scripts/run_codegen.py: WALL_CAP = cfg.sandbox.timeout*6.0 = 120s per-question via daemon thread + queue.get(timeout). Timeout -> fallback 0.0.',
  '',
  'RE-RUN RESULT (codegen_tidy2.log, cleaned up by user): [10/462] ok=0 fail=10 p95=120, [40/462] ok=5 fail=35 p95=120. ok near 0, all questions hit wall-cap 120s. API is NOT the issue (single 2.1s, 8-concurrent ~10s, no 429). The 462 pending questions were the HARD TAIL (slow ones still in-flight when first run was killed). wall-cap 120s too aggressive — cuts off solve() before it can return even a fallback. Daemon thread leaks on timeout.',
  '',
  'UNRESOLVED: phase-time solve() on a hard pending question (search vs build_cards vs codegen) to find the real bottleneck. Consider raising wall-cap to 180-240s OR making solve() time-aware (check deadline between phases, return fallback gracefully) instead of daemon-thread leak.',
  '',
  'CURRENT DATA STATE: results.jsonl (3.9MB, 1,012 records, ok=607/fail=405 per codegen_full.log 17:05, answer zero=640/nonzero=372). submission packaged once (validate ok=968/44 internal). results_tidy.jsonl deleted by user. No python process running.',
].join('\n')

phase('Synthesize')
const synthPrompt = 'You are rewriting D:\\GURU\\MEMORY.md to be a comprehensive, ACCURATE knowledge base for the GURU/ViFinQA project. The current MEMORY.md has stale claims; you must produce the corrected FULL content.\n' +
  '\n' +
  'You have:\n' +
  '1. The current MEMORY.md (Read it: ' + MEM + ').\n' +
  '2. Structured discrepancies from 6 verifiers (below) — each lists where current memory is wrong/stale/missing vs the CURRENT codebase (which already includes the start_line fixes).\n' +
  '3. OFFICIAL_SPEC — the authoritative BTC submission format (relevant_tables = report_id|<line>, NOT table_N).\n' +
  '4. CHANGES_TODAY — the ETL/catalog_builder start_line change, backfill script (ran), builder rewrite, smoke_fmt_100 script.\n' +
  '5. SESSION_FINDINGS — the codegen retry/timeout incident + 5 fixes + wall-cap issue.\n' +
  '\n' +
  'OUTPUT the COMPLETE, FINAL markdown content for D:\\GURU\\MEMORY.md. Vietnamese. Same 11-section structure (0..10) as the current file, but CORRECTED:\n' +
  '- Section 1: submission format per OFFICIAL_SPEC (relevant_tables = report_id|<line>).\n' +
  '- Section 2 (code map): update catalog_builder (start_line field, table_start_lines), builder (_load_start_lines, _table_ref_to_line), scripts (add backfill_start_line.py, smoke_fmt_100.py). Apply verifier discrepancies.\n' +
  '- Section 3 (pipeline): note builder rewrites relevant_tables to line format at packaging.\n' +
  '- Section 4 (schema): catalog_tables.csv now 16 cols incl start_line. relevant_tables format = report_id|<line>. csv_path filename still table_N.\n' +
  '- Section 5 (config): llm.timeout=30, retries=1, WALL_CAP=120.\n' +
  '- Section 6 (milestones + leaderboard): (A) TABLES_F2=0 was likely due to WRONG format (table_N not line); NOW FIXABLE — ETL records start_line, builder outputs report_id|<line>. Code fix DONE this session; pending rebuild+resubmit to test. Keep risk note (line-counting variant, no gold).\n' +
  '- Section 7: full session findings (codegen retry + wall-cap) verbatim-ish from SESSION_FINDINGS, PLUS a subsection "7b. relevant_tables format fix" summarizing CHANGES_TODAY.\n' +
  '- Section 8 (gotchas): correct relevant_tables to line format; note the 6/8 "fix" to table_N was wrong-direction; csv_path filename separate (table_N).\n' +
  '- Section 9 (next steps): update — (1) run smoke_fmt_100.py to verify format, (2) rebuild full submission (build_submission.py) with line-format relevant_tables, (3) submit, observe TABLES_F2 (expect >0 if line-counting matches BTC), (4) phase-time solve() for codegen wall-cap, (5) re-codegen subset with retry fixes, (6) M5 ReAct, (7) M7 dev-set, (8) submit high-quality subset instead of full 1012 (gold=506 only; avoid fallback-0.0 dragging average), (9) working notes paper. No dependencies on BTC answers — all decisions self-determined.\n' +
  '- Section 10: sources.\n' +
  '\n' +
  'RULES:\n' +
  '- Vietnamese. EXHAUSTIVE and PRECISE. Quote function names, file paths, config values, numbers. Do NOT invent.\n' +
  '- Apply EVERY verifier discrepancy (high+medium confidence must be fixed; low confidence note if unsure).\n' +
  '- Preserve correct details from current MEMORY.md; only change what is wrong/stale.\n' +
  '- Start with "# MEMORY.md — GURU · ViFinQA (Knowledge Base of Claude)". Output ONLY the file content (no fences around the whole thing, no commentary).\n' +
  '\n' +
  '=== VERIFIER DISCREPANCIES ===\n' +
  discrepancies.map((d) => '--- ' + d.verifier + ' ---\n' +
    (d.result.discrepancies || []).map((x) => '- [loc:' + x.location + '] claim:"' + x.claim + '" actual:"' + x.actual + '" fix:"' + x.fix + '" conf:' + x.confidence).join('\n') +
    '\nMISSING: ' + (d.result.missing || 'none')
  ).join('\n\n') +
  '\n\n=== OFFICIAL SPEC ===\n' + OFFICIAL_SPEC +
  '\n\n=== CHANGES TODAY ===\n' + CHANGES_TODAY +
  '\n\n=== SESSION FINDINGS ===\n' + SESSION_FINDINGS

const memory = await agent(synthPrompt, { label: 'synthesize:MEMORY.md', phase: 'Synthesize' })
return memory