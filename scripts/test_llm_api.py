"""Test gọi LLM qua OpenRouter 1 lần — xác minh model qwen/qwen3.5-9b hoạt động.

Chạy:  $env:OPENROUTER_API_KEY="sk-or-..."  rồi
        .venv\\Scripts\\python scripts/test_llm_api.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openai import OpenAI  # noqa: E402

from vifinqa.config import Config  # noqa: E402


def main() -> int:
    cfg = Config.load(ROOT / "configs" / "api.yaml")
    api_key = cfg.llm.effective_api_key()
    if not api_key:
        print("❌ Thiếu API key. Set env OPENROUTER_API_KEY rồi chạy lại.")
        return 1

    client = OpenAI(
        base_url=cfg.llm.base_url,
        api_key=api_key,
        timeout=cfg.llm.timeout,
        max_retries=cfg.llm.retries,
        default_headers=cfg.llm.extra_headers or None,
    )

    print(f"Gọi {cfg.llm.model_id} qua {cfg.llm.base_url} ...")
    resp = client.chat.completions.create(
        model=cfg.llm.model_id,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
        messages=[
            {
                "role": "user",
                "content": "Trả lời ngắn gọn: 2 + 3 * 4 bằng bao nhiêu?",
            }
        ],
    )
    print("✅ Model phản hồi:")
    print(resp.choices[0].message.content)
    print(f"   usage: {resp.usage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
