"""llm.py — LLM client Text-to-Pandas (Qwen3.5-9B qua OpenRouter).

`generate_query(messages)` → gọi chat.completions, strip reasoning tokens Qwen3
(thinking mặc định), trích code trong fence ```python ... ```. Retry lỗi tạm thời
(429/5xx/timeout) với backoff mũ. Trả chuỗi pandas_query (rỗng nếu parse thất bại).

Reuse cfg.llm (OpenAI client, base_url, model_id, temperature, max_tokens, timeout).
"""

from __future__ import annotations

import ast
import random
import re
import time

from openai import OpenAI, APIStatusError, APITimeoutError, APIConnectionError

from vifinqa.config import Config

# Strip khối think của Qwen3 (inline) — an toàn nếu content không có.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Code fence ```python ... ``` (hoặc ``` ... ```).
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
# Dòng import — LLM hay tự thêm `import pandas as pd` dù đã cấm; strip vì sandbox
# đã inject pd/np/math/re/json sẵn.
_IMPORT_LINE_RE = re.compile(r"^\s*(?:import\s+\w|from\s+\w[\w.]*\s+import\s+)")


def _strip_imports(code: str) -> str:
    """Bỏ các dòng import (sandbox đã inject pd/np/...) — LLM thường quên quy tắc."""
    out = []
    for line in code.splitlines():
        if _IMPORT_LINE_RE.match(line):
            continue
        out.append(line)
    return "\n".join(out).strip()


# Block `def vn_num(...)` (có thể kèm decorator) — sandbox đã inject vn_num sẵn;
# LLM hay định nghĩa lại (sai logic VN) → strip để dùng bản inject.
_VN_NUM_DEF_RE = re.compile(r"^\s*(?:@\w+\s*\n\s*)?def\s+vn_num\s*\(", re.MULTILINE)


def _strip_vn_num_def(code: str) -> str:
    """Bỏ định nghĩa `def vn_num(...): ...` (giữ bản inject sẵn trong sandbox)."""
    m = _VN_NUM_DEF_RE.search(code)
    if not m:
        return code
    lines = code.splitlines(keepends=True)
    # tìm dòng chứa def
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"\s*(?:@\w+\s*$)?\s*def\s+vn_num\s*\(", ln):
            start = i
            # bao gồm decorator dòng trước nếu có
            if i > 0 and lines[i - 1].strip().startswith("@"):
                start = i - 1
            break
    if start is None:
        return code
    # bỏ dòng def + toàn bộ dòng thụt tiếp theo (cho đến dòng không thụt & không rỗng)
    end = start + 1
    while end < len(lines):
        ln = lines[end]
        if ln.strip() == "":
            end += 1
            continue
        if ln[0].isspace():  # thụt lề → vẫn thuộc body def
            end += 1
            continue
        break
    remaining = lines[:start] + lines[end:]
    return "\n".join("".join(remaining).splitlines()).strip()


def _is_synthetic_df(value: ast.AST) -> bool:
    """Vế phải tạo DataFrame giả (pd.DataFrame/pd.concat/dict/list/np.array...)."""
    if isinstance(value, (ast.Dict, ast.List, ast.Set, ast.Tuple)):
        return True
    if isinstance(value, ast.Call):
        fn = value.func
        if isinstance(fn, ast.Name) and fn.id in {"DataFrame", "array", "concat", "Series"}:
            return True
        if isinstance(fn, ast.Attribute) and fn.attr in {"DataFrame", "array", "concat", "Series"}:
            return True
    return False


def _strip_df_reassign(code: str) -> str:
    """Bỏ các lệnh `dfN = <data giả>` (LLM hay tự tạo DataFrame thay vì dùng df có sẵn).

    Chỉ bỏ khi vế phải tạo data MỚI (pd.DataFrame/dict/list/...). Giữ nguyên
    `df1 = df1[...]` (transform hợp lệ) và biến khác (`sub = ...`).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    body = []
    removed = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            is_df = all(isinstance(t, ast.Name) and re.fullmatch(r"df\d+", t.id) for t in targets)
            if is_df and _is_synthetic_df(node.value):
                removed = True
                continue
        body.append(node)
    if not removed:
        return code
    tree.body = body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _extract_code(text: str) -> str:
    """Trích code pandas từ output LLM (strip think, lấy fence cuối, bỏ import/def giả)."""
    if not text:
        return ""
    cleaned = _THINK_RE.sub("", text).strip()
    fences = _FENCE_RE.findall(cleaned)
    if fences:
        # Lấy fence cuối (LLM có thể kèm giải thích trước code).
        return _strip_df_reassign(_strip_vn_num_def(_strip_imports(fences[-1].strip())))
    # Không có fence: nếu toàn văn nhìn như code (có `result =`), trả nguyên.
    if "result" in cleaned and "\n" in cleaned:
        return _strip_df_reassign(_strip_vn_num_def(_strip_imports(cleaned.strip())))
    return ""


class LLMClient:
    """Client gọi LLM codegen — 1 instance dùng cho cả batch (thread-safe)."""

    def __init__(self, cfg: Config):
        llm = cfg.llm
        self.model = llm.model_id
        self.temperature = llm.temperature
        self.max_tokens = llm.max_tokens
        self.timeout = llm.timeout
        self.thinking = llm.thinking
        # SDK max_retries=1: chỉ cho SDK thử lại nhanh 1 lần ở transport, để manual
        # loop dưới đây là layer retry duy nhất (tránh double-retry chồng nhau gây kẹt
        # 4×60s+15s backoff mỗi generate_query).
        self._client = OpenAI(
            base_url=llm.base_url,
            api_key=llm.effective_api_key(),
            timeout=llm.timeout,
            max_retries=1,
            default_headers=llm.extra_headers or None,
        )

    def generate_query(self, messages: list[dict], max_retries: int = 2) -> str:
        """Gọi LLM → trả pandas_query string (rỗng nếu lỗi/hỏng)."""
        # Qwen3 reasoning plugin: tắt thinking (cfg.llm.thinking=False mặc định) để
        # tránh reasoning ăn hết max_tokens → content rỗng (overflow đã thấy 4258 tok).
        extra_body = {"reasoning": {"enabled": self.thinking}} if self.thinking is not None else None
        last_err = ""
        for attempt in range(max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=messages,
                    extra_body=extra_body,
                )
                content = resp.choices[0].message.content or ""
                return _extract_code(content)
            except (APITimeoutError, APIConnectionError) as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt == max_retries - 1:
                    break
                time.sleep(2.0 + random.uniform(0, 1))
            except APIStatusError as e:
                last_err = f"HTTP {e.status_code}: {e}"
                if attempt == max_retries - 1 or e.status_code not in (429, 500, 502, 503, 504):
                    break
                time.sleep(2.0 + random.uniform(0, 1))
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                break
        return ""