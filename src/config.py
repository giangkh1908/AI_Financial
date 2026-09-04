"""
Central Configuration Module.
Single source of truth for all file paths, database URIs, model configs, and service ports.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Storage & Data Files
DB_PATH = DATA_DIR / "financial.db"
CODE_STOCK_PATH = DATA_DIR / "code_stock.csv"
QUESTIONS_PATH = DATA_DIR / "questions" / "questions.jsonl"
STATEMENTS_DIR = DATA_DIR / "financial_statements"

# Model & Serving Config
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
SLM_MODEL_NAME = os.getenv("SLM_MODEL_NAME", "qwen3.5-4b-sql")

# REST API Config
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Execution & Safety Guards
SQL_TIMEOUT_SECONDS = 5.0
DEFAULT_ROW_LIMIT = 10
