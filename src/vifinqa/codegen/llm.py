"""llm.py — LLM client Text-to-Pandas (Qwen3.5-9B qua OpenRouter).

`generate_query(messages)` → gọi chat.completions, strip reasoning tokens Qwen3
(thinking mặc định), trích code trong fence ```python ... ```. Retry lỗi tạm thời
(429/5xx/timeout) với backoff mũ. Trả chuỗi pandas_query (rỗng nếu parse thất bại).

Reuse cfg.llm (OpenAI client, base_url, model_id, temperature, max_tokens, timeout).
"""

from __future__ import annotations

import ast
import io
import random
import re
import time
import tokenize

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
            is_df = all(isinstance(t, ast.Name) and re.fullmatch(r"df\d*", t.id) for t in targets)
            if is_df and _is_synthetic_df(node.value):
                removed = True
                continue
        body.append(node)
    if not removed:
        return code
    tree.body = body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


class _CommentStripper:
    """Bỏ comment (`# ...`) nhưng giữ `#` bên trong string literal.

    Rebuild theo dòng (không dùng untokenize — untokenize giữ nguyên khoảng trắng cũ
    sau khi xoá comment → chèn khoảng trắng khổng lồ, phá indent/syntax). Giữ nguyên
    leading whitespace (indent), bỏ comment full-line và trailing comment, bỏ dòng rỗng.
    Dùng `tokenize` per-line để không phá chuỗi literal chứa `#`.
    """

    @staticmethod
    def strip(code: str) -> str:
        if not code:
            return code
        out: list[str] = []
        for line in code.splitlines():
            if not line.strip():
                continue  # bỏ dòng rỗng (không cần thiết về mặt cú pháp)
            if line.lstrip().startswith("#"):
                continue  # bỏ nguyên dòng comment
            try:
                toks = list(tokenize.generate_tokens(io.StringIO(line).readline))
                cut = -1
                for t in toks:
                    if t.type == tokenize.COMMENT:
                        cut = t.start[1]
                        break
                if cut >= 0:
                    line = line[:cut].rstrip()
            except Exception:
                pass  # tokenize lỗi (cú pháp cắt giữa) → giữ nguyên dòng
            if line.strip():
                out.append(line)
        return "\n".join(out)


def _extract_code(text: str) -> str:
    """Trích code pandas từ output LLM (strip think, lấy fence cuối, bỏ import/def giả)."""
    if not text:
        return ""
    # Sanitize curly quotes/smart quotes → straight quotes (deepseek-coder 1.3B hay lỗi này)
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    cleaned = _THINK_RE.sub("", text).strip()
    fences = _FENCE_RE.findall(cleaned)
    if fences:
        # Lấy fence cuối (LLM có thể kèm giải thích trước code).
        return _CommentStripper.strip(
            _strip_df_reassign(_strip_vn_num_def(_strip_imports(fences[-1].strip())))
        )
    # Không có fence: nếu toàn văn nhìn như code (có `result =`), trả nguyên.
    if "result" in cleaned and "\n" in cleaned:
        return _CommentStripper.strip(
            _strip_df_reassign(_strip_vn_num_def(_strip_imports(cleaned.strip())))
        )
    return ""


class LLMClient:
    """Client gọi LLM codegen — 1 instance dùng cho cả batch (thread-safe).

    section='llm'         → head LLM (Qwen3.5-9B, phân tích câu hỏi)
    section='codegen_llm' → pandas expert (deepseek-coder 1.3B)

    Hỗ trợ fallback chain: primary (cfg.llm / cfg.codegen_llm) + llm.fallbacks.
    Local-first: thử sticky (provider lần trước thành công) trước; transient
    (timeout/conn/429/5xx) → thử provider kế tiếp. Provider fail toàn bộ → vào
    cooldown 30s (tránh đập endpoint chết mỗi query → phí 1 timeout/provider).
    """

    _COOLDOWN = 30.0  # giây — provider fail toàn bộ bị bỏ qua trong khoảng này

    def __init__(self, cfg: Config, section: str = "llm"):
        if section == "codegen_llm" and cfg.codegen_llm is not None:
            primary = cfg.codegen_llm
        else:
            primary = cfg.llm
        # specs: primary (LLMConfig) + fallbacks (ProviderRef) — cùng shape cần thiết
        specs = [primary, *primary.fallbacks]
        self.temperature = primary.temperature
        self.max_tokens = primary.max_tokens
        self._primary_timeout = primary.timeout
        self._primary_thinking = primary.thinking
        self._clients: list[OpenAI] = []
        self._models: list[str] = []
        self._providers: list[str] = []
        self._thinkings: list[bool] = []
        for s in specs:
            # timeout: ProviderRef.timeout=None → kế thừa primary; LLMConfig.timeout=float
            t = getattr(s, "timeout", None)
            t = self._primary_timeout if t is None else t
            # thinking: ProviderRef.thinking=None → kế thừa primary
            th = s.thinking if s.thinking is not None else self._primary_thinking
            # SDK max_retries=1: chỉ thử lại nhanh 1 lần ở transport; manual loop dưới là
            # layer retry duy nhất (tránh double-retry chồng nhau kẹt 4×60s+15s).
            self._clients.append(OpenAI(
                base_url=s.base_url,
                api_key=s.effective_api_key(),
                timeout=t,
                max_retries=1,
                default_headers=s.extra_headers or None,
            ))
            self._models.append(s.model_id)
            self._providers.append(s.provider)
            self._thinkings.append(bool(th))
        self._sticky = 0
        self._failed_until: dict[int, float] = {}

    def _order(self) -> list[int]:
        """Thứ tự thử: sticky trước (nếu không cooldown), rồi các provider khác còn sống."""
        now = time.time()
        alive = [i for i in range(len(self._clients)) if self._failed_until.get(i, 0.0) <= now]
        if not alive:  # tất cả trong cooldown → thử hết (không bỏ im)
            alive = list(range(len(self._clients)))
        if self._sticky in alive:
            return [self._sticky] + [i for i in alive if i != self._sticky]
        return alive

    def _try_provider(self, idx: int, messages: list[dict], extract_code: bool, max_retries: int) -> str | None:
        """Thử 1 provider (in-provider retry ≤ max_retries). Trả content/"" hoặc None nếu provider fail."""
        client = self._clients[idx]
        model = self._models[idx]
        provider = self._providers[idx]
        thinking = self._thinkings[idx]
        # reasoning/thinking theo provider:
        # - vllm: chat_template_kwargs.enable_thinking (Qwen3 chat template) — tắt server-side đúng cách.
        # - deepinfra/openrouter: plugin reasoning (extra_body riêng của chúng).
        # - khác: không gửi extra_body lạ.
        if provider == "vllm":
            extra_body = {"chat_template_kwargs": {"enable_thinking": thinking}}
        elif provider in ("deepinfra", "openrouter"):
            extra_body = {"reasoning": {"enabled": thinking}}
        else:
            extra_body = None
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=messages,
                    extra_body=extra_body,
                )
                content = resp.choices[0].message.content or ""
                self._sticky = idx  # thành công → sticky nhảy về provider này
                return _extract_code(content) if extract_code else content
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
        print(f"[llm] provider#{idx} {provider} {model} fail → {last_err}; thử provider kế", flush=True)
        self._failed_until[idx] = time.time() + self._COOLDOWN
        return None

    def _call_chain(self, messages: list[dict], extract_code: bool, max_retries: int = 2) -> str:
        """Thử chain sticky-first; provider nào thành công trả liền. Hết chain → ""."""
        for idx in self._order():
            out = self._try_provider(idx, messages, extract_code, max_retries)
            if out is not None:
                return out
        return ""

    def generate_raw(self, messages: list[dict], max_retries: int = 2) -> str:
        """Gọi LLM → trả text thô (dùng cho table-selector / phân tích).

        Không extract code — giữ nguyên response để parse JSON.
        """
        return self._call_chain(messages, extract_code=False, max_retries=max_retries)

    def generate_query(self, messages: list[dict], max_retries: int = 2) -> str:
        """Gọi LLM → trả pandas_query string (rỗng nếu lỗi/hỏng)."""
        return self._call_chain(messages, extract_code=True, max_retries=max_retries)