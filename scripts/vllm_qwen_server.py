"""vllm_qwen_server.py — 1 file chạy trên GPU thuê, serving Qwen3.5-9B qua vLLM
(OpenAI-compatible API).

Cách dùng (GPU thuê — Vast/RunPod, template PyTorch, ≥16GB VRAM lý tưởng 24GB):
    python vllm_qwen_server.py                         # tự cài vllm + serve
    python vllm_qwen_server.py --port 8001 --gpu-memory-fraction 0.65
    python vllm_qwen_server.py --model Qwen/Qwen3.5-9B --max-model-len 8192

API (OpenAI-compatible, vLLM built-in):
    POST /v1/chat/completions    (chat — codegen dùng cái này)
    GET  /health

Chạy kèm bge_m3_server.py (port 8000, embed) trên CÙNG GPU:
    1) python scripts/bge_m3_server.py            # port 8000, chiếm ~10-20% VRAM
    2) python scripts/vllm_qwen_server.py         # port 8001, gpu-memory-fraction 0.65 (60-70%)
    Còn ~15% headroom. GPU yếu/OOM → giảm --gpu-memory-fraction xuống 0.5 hoặc --max-model-len.

Thinking (Qwen3 reasoning) TẮT: server không bật --enable-reasoning; client (LLMClient)
gửi chat_template_kwargs.enable_thinking=False qua extra_body → Qwen3 chat template
bỏ khối 'think' của Qwen3, tránh reasoning ăn hết max_tokens.

Local (máy của bạn): đổi configs/local_vllm.yaml
    llm.base_url: http://<GPU_IP>:8001/v1
    llm.api_key:  <giống --api-key khi khởi động, mặc định 'vllm'>
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

MODEL = "Qwen/Qwen3.5-9B"


def _pip_install(pkgs: list[str]) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)


def _detect_cuda_driver() -> str | None:
    import re

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        m = re.search(r"(\d+)\.(\d+)", out)
        return f"{m.group(1)}.{m.group(2)}" if m else None
    except Exception:
        return None


def _torch_cuda_ok() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _ensure_vllm() -> None:
    try:
        import vllm  # noqa: F401

        return
    except ImportError:
        pass
    driver = _detect_cuda_driver()
    print(f"[deps] CUDA driver: {driver}")
    # vLLM wheel kéo đúng torch khớp CUDA; cài bản ổn định mới.
    _pip_install(["vllm"])
    try:
        import vllm

        print(f"[deps] vllm installed: {getattr(vllm, '__version__', '?')}")
    except Exception as e:
        print(f"[deps] ⚠️ vllm import lỗi sau cài: {e}")
    if not _torch_cuda_ok():
        print("[deps] ⚠️ CUDA không khả dụng — vLLM cần GPU. Kiểm tra nvidia-smi / mount /dev/nvidia* / driver đủ mới.")


def _print_yaml(host: str, port: int, api_key: str, model: str) -> None:
    print("\n================ Copy vào configs/local_vllm.yaml (llm:) ================")
    print(f"""llm:
  provider: vllm
  base_url: http://{host}:{port}/v1
  api_key: {api_key}
  model_id: {model}
  temperature: 0.0
  max_tokens: 4096
  timeout: 60.0
  thinking: false
  fallbacks:
    - provider: deepinfra
      base_url: https://api.deepinfra.com/v1/openai
      api_key: ""   # → env DEEPINFRA_TOKEN
      model_id: Qwen/Qwen3.5-9B
      thinking: false
=========================================================================\n""")


def main() -> int:
    ap = argparse.ArgumentParser(description="vLLM serving Qwen3.5-9B — chạy trên GPU thuê")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--served-name", default="", help="tên model lộ ra API (mặc định = --model)")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-fraction", type=float, default=0.65,
                    help="60-70%% VRAM cho Qwen3.5-9B; bge_m3_server dùng ~10-20%% còn lại")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "vllm"),
                    help="API key yêu cầu; client phải gửi trùng (mặc định 'vllm').")
    ap.add_argument("--extra", default="", help="thêm args truyền thẳng cho vllm serve (vd '--enforce-eager')")
    args = ap.parse_args()

    _ensure_vllm()

    served = args.served_name or args.model
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--served-model-name", served,
        "--host", args.host,
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-fraction", str(args.gpu_memory_fraction),
        "--dtype", args.dtype,
        "--api-key", args.api_key,
        # thinking OFF: KHÔNG thêm --enable-reasoning. Client gửi chat_template_kwargs
        # enable_thinking=False → Qwen3 template bỏ khối think.
    ]
    if args.extra:
        cmd += args.extra.split()

    print(f"[serve] {' '.join(cmd)}")
    print(f"[serve] http://{args.host}:{args.port} | gpu-mem-fraction={args.gpu_memory_fraction}")
    print("[serve] ⚠️ chạy bge_m3_server.py (port 8000) TRƯỚC nếu cùng GPU.")
    _print_yaml(args.host, args.port, args.api_key, args.model)

    try:
        proc = subprocess.run(cmd)
        return proc.returncode
    except KeyboardInterrupt:
        print("\n[serve] Ctrl-C — dừng vLLM")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())