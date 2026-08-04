"""Hằng số toàn cục — các giá trị dùng chung toàn pipeline."""

# Đơn vị tiền tệ (Việt Nam) → hệ số VND
# Quy ước: value_vnd = giá_trị_số * UNIT_FACTORS[đơn_vị]
UNIT_FACTORS: dict[str, float] = {
    "đồng": 1.0,
    "VND": 1.0,
    "nghìn": 1e3,
    "triệu": 1e6,
    "tỷ": 1e9,
}

# Tolerance đáp án tuyệt đối (giả định theo paper ViFinQA; BTC chưa công bố chính thức)
ANSWER_ABS_TOL = 0.01

# ReAct loop
STEP_BUDGET = 10          # số bước action tối đa mỗi câu
PARSE_FAIL_LIMIT = 2      # lỗi parse JSON liên tiếp → buộc finalize
RUN_PANDAS_RETRY = 2      # số lần retry khi run_pandas crash

# Sandbox
SANDBOX_TIMEOUT = 20      # giây
MAX_CODE_LEN = 4000       # độ dài tối đa pandas_query
MAX_AST_NODES = 300       # số node AST tối đa

# Retrieval
K = 10                    # top-k bảng trả cho agent
RERANK_DEPTH = 100        # số bảng đưa vào reranker
RRF_K = 60                # hằng số K trong RRF fusion

# Loại báo cáo
REPORT_TYPES = ("consolidated", "separate", "aggregated", "other")

# Loại bảng lõi (Tier A facts)
STATEMENT_TYPES = ("balance_sheet", "income", "cash_flow")
