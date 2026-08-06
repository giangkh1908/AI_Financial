export const meta = {
  name: 'verify-memory-md',
  description: 'Verify MEMORY.md claims against actual GURU codebase + data, return structured discrepancies for fixing',
  phases: [
    { title: 'Verify', detail: 'parallel verifiers per project slice check claims vs actual files' },
    { title: 'Collect', detail: 'aggregate all discrepancies' },
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
          location: { type: 'string', description: 'section + approximate quote/location in MEMORY.md' },
          claim: { type: 'string', description: 'what MEMORY.md asserts' },
          actual: { type: 'string', description: 'what the actual codebase/data shows' },
          fix: { type: 'string', description: 'concrete edit instruction to correct MEMORY.md' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
        required: ['location', 'claim', 'actual', 'fix', 'confidence'],
      },
    },
    missing: { type: 'string', description: 'important codebase facts NOT captured in MEMORY.md, or "none"' },
  },
  required: ['verifier', 'discrepancies', 'missing'],
}

const VERIFIERS = [
  {
    key: 'etl',
    prompt: 'You are verifying the ETL-related claims in ' + MEM + ' against the actual codebase.\n' +
      'Read ' + MEM + ' (sections 2 ETL map, 3 ETL flow, 8 ETL gotchas, 4 derived outputs for ETL) then verify EVERY concrete claim against the actual files:\n' +
      '- ' + ROOT + '\\src\\vifinqa\\etl\\numbers.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\etl\\parser.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\etl\\statements.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\etl\\catalog_builder.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\etl\\facts_builder.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\etl\\tidy.py\n' +
      'Check: function names/signatures exist, regex names correct, constants (UNIT_FACTORS, _VAS_CODE_RE, etc.) correct, behaviors (colspan expand yes / rowspan no, parse_vn_number("4.037")=4037, detect_unit anchor-bounded, find_item_code_col >=70% >=3 rows, emit_facts dedup, wide_to_tidy parse VN only, etc.). Also count rows in ' + ROOT + '\\data\\derived\\catalog_tables.csv and facts_all.csv (Bash wc -l) to verify 146,246 / 10,797 / 377,578 numbers.\n' +
      'Return ONLY the structured object. For each discrepancy: location, claim (what memory says), actual (what code shows), fix (concrete edit), confidence. If a claim is correct, do NOT list it. Put codebase facts missing from memory in "missing".',
  },
  {
    key: 'retrieval',
    prompt: 'Verify retrieval claims in ' + MEM + ' against actual files.\n' +
      'Read ' + MEM + ' (sections 2 retrieval, 3 retrieval flow, 8 qdrant/retrieval gotchas) then check against:\n' +
      '- ' + ROOT + '\\src\\vifinqa\\retrieval\\entity.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\retrieval\\index.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\retrieval\\search.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\retrieval\\facts_index.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\retrieval\\pipeline.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\retrieval\\rerank.py\n' +
      'Check: function/class names, Entities dataclass fields, extract_tickers 4-stage, relevant_tables_key() format report_id|table_N, FusionQuery(Fusion.RRF), point_id uuid5, _SPARSE_BITS=21, statement_bonus value (base.yaml vs api.yaml), embed_statement_only, rerank disabled + _DTYPE_KWARG bug, facts_for_table exact match, facts_all.csv no report_id column.\n' +
      'Verify Qdrant point count claim 118,728: run Bash `python -c "from qdrant_client import QdrantClient; c=QdrantClient(url=\\\"http://localhost:6333\\\"); print(c.count(\\\"bctc_tables\\\"))"` if Docker running, else read ' + ROOT + '\\data\\out\\rebuild_index.log. Return ONLY structured object.',
  },
  {
    key: 'codegen-sandbox',
    prompt: 'Verify codegen + sandbox claims in ' + MEM + ' against actual files.\n' +
      'Read ' + MEM + ' (sections 2 codegen+sandbox, 3 codegen+sandbox flow, 8 sandbox contract + codegen contract) then check against:\n' +
      '- ' + ROOT + '\\src\\vifinqa\\codegen\\prompt.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\codegen\\llm.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\sandbox\\ast_check.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\sandbox\\runner.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\sandbox\\executor.py\n' +
      'Check CRITICAL contract details: (1) runner injects dict `dfs` keyed by table_ref (not bare df1/df2) + `df` alias when 1 CSV; (2) read_csv dtype=str, keep_default_na=False, index_col=None, encoding; (3) vn_num inlined in runner; (4) _SAFE_BUILTINS; (5) generate_query max_retries=2 (post-fix), backoff 2.0s; (6) OpenAI(max_retries=1) hardcoded; (7) _extract_code strips think/fence/imports/def vn_num/dfN reassign; (8) thinking=False extra_body; (9) ast_check default max_code_len/max_ast_nodes actual values (memory says 4000/300 but base.yaml says 8000/800 — which wins?); (10) prompt teaches dfs[\"df1\"] or dfs[\"<table_ref>\"] — memory is inconsistent, check actual prompt.py.\n' +
      'Return ONLY structured object with every discrepancy.',
  },
  {
    key: 'agent-submission',
    prompt: 'Verify agent + submission claims in ' + MEM + ' against actual files.\n' +
      'Read ' + MEM + ' (sections 2 agent+submission, 3 codegen flow + submission flow, 8 submission gotchas) then check against:\n' +
      '- ' + ROOT + '\\src\\vifinqa\\agent\\loop.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\submission\\builder.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\submission\\validate.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\submission\\pack.py\n' +
      'Check: solve() signature + max_retries default (1 post-fix), _build_table_card, _ensure_tidy path derived/evidence/{rid}__{tid}.csv, _MAX_SAMPLE_ROWS=8, _MAX_FACT_HINTS=25, _repair, _make_record evidence variable=df{i+1} csv_path=data/{rid}__{tid}.csv, builder _VN_NUM_DEF + _with_vn_num + tidy regenerate fallback, validate tol 0.01 short-circuit 0.0 ThreadPool 8, _table_ref_from_csv_path, pack ZIP root 1 json.\n' +
      'Return ONLY structured object.',
  },
  {
    key: 'config-scripts-tests',
    prompt: 'Verify config + loader + constants + scripts + tests claims in ' + MEM + ' against actual files.\n' +
      'Read ' + MEM + ' (sections 2 core + scripts + tests, 5 config + knobs, 8 checkpoint gotchas) then check against:\n' +
      '- ' + ROOT + '\\src\\vifinqa\\config.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\loader.py\n' +
      '- ' + ROOT + '\\src\\vifinqa\\constants.py\n' +
      '- ' + ROOT + '\\configs\\base.yaml\n' +
      '- ' + ROOT + '\\configs\\api.yaml\n' +
      '- ' + ROOT + '\\scripts\\run_codegen.py\n' +
      '- ' + ROOT + '\\scripts\\build_submission.py\n' +
      '- ' + ROOT + '\\scripts\\run_etl.py\n' +
      '- ' + ROOT + '\\scripts\\run_facts.py\n' +
      '- ' + ROOT + '\\scripts\\run_retrieval.py\n' +
      'Glob ' + ROOT + '\\tests\\*.py and verify test file list + purposes.\n' +
      'Check: all config knob values (llm.timeout=30, retries=1, max_tokens=4096, sandbox.timeout=20, max_code_len=8000, max_ast_nodes=800, k=10, rerank_depth=100, statement_bonus=0.001, embed_statement_only=false, rerank.enabled=false, workers=4), WALL_CAP = cfg.sandbox.timeout*6.0 = 120 in run_codegen.py, constants.py diverge values, deep-merge base+api, _load_done checkpoint, lock_file unimplemented.\n' +
      'Return ONLY structured object.',
  },
  {
    key: 'data-state',
    prompt: 'Verify data-state + results + submission claims in ' + MEM + ' against actual disk state.\n' +
      'Read ' + MEM + ' (section 4 schema/outputs, 6 status, 7 current state) then check actual disk:\n' +
      '- Bash: ls -la ' + ROOT + '\\data\\derived\\ and ' + ROOT + '\\data\\out\\\n' +
      '- Bash: wc -l ' + ROOT + '\\data\\derived\\catalog_tables.csv ' + ROOT + '\\data\\derived\\facts_all.csv ' + ROOT + '\\data\\derived\\documents.csv ' + ROOT + '\\data\\out\\results.jsonl\n' +
      '- Read ' + ROOT + '\\data\\derived\\etl_state.json, retrieval_state.json, facts_state.json (whichever exist)\n' +
      '- Read first line of catalog_tables.csv and facts_all.csv and documents.csv (headers)\n' +
      '- Read ' + ROOT + '\\data\\out\\codegen_full.log fully\n' +
      '- Read ' + ROOT + '\\data\\out\\rebuild_index.log\n' +
      '- Read ' + ROOT + '\\data\\out\\_fts.txt and _fts2.txt\n' +
      '- Check submission: Bash ls -la ' + ROOT + '\\data\\out\\submission\\ and read validation.jsonl summary if exists, count files in submission\\data\\ (Bash: ls data\\out\\submission\\data | wc -l)\n' +
      '- Read ' + ROOT + '\\data\\derived\\retrieval_metrics.json if exists\n' +
      '- Verify claims: results.jsonl 1,012 records ok=607 fail=405 (compute from results.jsonl: count _ok true/false, answer zero vs nonzero), submission validate ok=968/44, evidence 7,469 tidy CSV rebuilt 6/8 17:33, qdrant 118,728 points, evidence dir actual count.\n' +
      'Return ONLY structured object. Be precise with actual numbers.',
  },
]

phase('Verify')
const results = await parallel(
  VERIFIERS.map((v) => () => agent(v.prompt, { label: 'verify:' + v.key, phase: 'Verify', schema: SCHEMA }))
)
const out = VERIFIERS.map((v, i) => ({ verifier: v.key, result: results[i] })).filter((x) => x.result)
return out