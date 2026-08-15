"""serve_all.py — 1 lệnh khởi động cả 2 model local trên cùng GPU thuê.

Spawn bge_m3_server.py (embed BGE-M3, port 8000) và vllm_qwen_server.py (LLM Qwen3.5-9B,
port 8001) làm 2 process con, đúng thứ tự:
  1) bge TRƯỚC — chiếm ~10-20% VRAM (model nhỏ, load nhanh).
  2) poll /health port 8000 tới khi ready (≈ model load xong).
  3) vLLM SAU — lấy 65% VRAM (--gpu-memory-fraction 0.65), còn ~15% headroom.
Bge phải lên trước vLLM vì cùng GPU — tránh tranh VRAM lúc load 2 model cùng lúc.

Cách dùng (GPU thuê, ≥16GB VRAM lý tưởng 24GB):
    python scripts/serve_all.py
    python scripts/serve_all.py --host 0.0.0.0 --embed-port 8000 --llm-port 8001
    python scripts/serve_all.py --gpu-memory-fraction 0.5     # GPU yếu/OOM
    python scripts/serve_all.py --no-llm                      # chỉ chạy embed

Khi 2 port sẵn sàng, in ra block YAML cho configs/local_vllm.yaml (cả llm + retrieval.embedding).
Ctrl-C → tắt sạch cả 2 process group (vLLM grandchild cũng chết, không leak GPU).

Lần đầu chậm: bge tự cài torch+FlagEmbedding + tải BAAI/bge-m3 (~2GB), vLLM tự cài vllm
+ tải Qwen/Qwen3.5-9B (~18GB). Lần sau chỉ load (~30s bge + ~60s vLLM).

Trên máy local: GPU ở remote thì đổi base_url http://localhost → http://<GPU_IP> trong
configs/local_vllm.yaml. Cùng máy thì giữ localhost.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# Console Windows (cp1252) không in được tiếng Việt → ép UTF-8 (giống build_retrieval_index.py).
# Cũng giúp forward output con chứa tiếng Việt không bị лом cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

ROOT = Path(__file__).resolve().parents[1]
EMBED_SCRIPT = ROOT / "scripts" / "bge_m3_server.py"
LLM_SCRIPT = ROOT / "scripts" / "vllm_qwen_server.py"


def _wait_health(url: str, name: str, timeout: float) -> bool:
    """Poll GET url tới 200 (vLLM /health trả 503 khi đang load → bỏ qua). Trả True nếu ready."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    print(f"[{name}] ✅ ready sau {time.time()-t0:.0f}s — {url}", flush=True)
                    return True
        except urllib.error.HTTPError as e:
            # 503 = đang load (vLLM); lỗi khác cũng cứ poll tiếp
            if e.code not in (502, 503, 404):
                print(f"[{name}] HTTP {e.code} — vẫn poll", flush=True)
        except Exception:
            pass  # Connection refused khi process chưa lên → poll tiếp
        time.sleep(3)
    print(f"[{name}] ⚠️ KHÔNG ready sau {timeout:.0f}s — {url}", file=sys.stderr, flush=True)
    return False


def _pipe(proc: subprocess.Popen, tag: str) -> None:
    """Forward stdout của proc ra parent với prefix (vd '[embed] [model] ...')."""
    def _p(stream, tag):
        try:
            for line in iter(stream.readline, ""):
                sys.stdout.write(f"{tag} {line}")
                sys.stdout.flush()
        finally:
            try:
                stream.close()
            except Exception:
                pass

    if proc.stdout:
        threading.Thread(target=_p, args=(proc.stdout, tag), daemon=True).start()
    if proc.stderr:
        threading.Thread(target=_p, args=(proc.stderr, tag), daemon=True).start()


def _kill_group(proc: subprocess.Popen, grace: float = 15.0) -> None:
    """Tắt process con + cả group (vLLM grandchild). SIGINT trước (graceful), rồi SIGKILL."""
    if proc.poll() is not None:
        return
    if os.name == "posix":
        # start_new_session → proc.pid là pgid của group mới (gồm vLLM grandchild)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
    else:  # Windows: taskkill /T kill cả cây process
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            capture_output=True, timeout=20)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _print_yaml(host: str, embed_port: int, llm_port: int, api_key: str, model: str) -> None:
    # host 0.0.0.0 → client cần IP thật; gợi ý localhost khi cùng máy.
    client_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    print("\n================ Copy vào configs/local_vllm.yaml ================")
    print(f"""llm:
  provider: vllm
  base_url: http://{client_host}:{llm_port}/v1
  api_key: {api_key}
  model_id: {model}
  thinking: false
  fallbacks:
    - provider: deepinfra
      base_url: https://api.deepinfra.com/v1/openai
      api_key: ""   # → env DEEPINFRA_TOKEN
      model_id: {model}
      thinking: false

codegen_llm:
  provider: vllm
  base_url: http://{client_host}:{llm_port}/v1
  api_key: {api_key}
  model_id: {model}
  fallbacks:
    - provider: deepinfra
      base_url: https://api.deepinfra.com/v1/openai
      api_key: ""
      model_id: {model}

retrieval:
  embedding:
    provider: ngrok            # HTTP /embed tới bge_m3_server
    base_url: http://{client_host}:{embed_port}
    model: baai/bge-m3
    dense_dim: 1024
    fallbacks:
      - provider: deepinfra
        base_url: https://api.deepinfra.com/v1/openai
        api_key: ""
        model: BAAI/bge-m3
        dense_dim: 1024
========================================================================\n""")


def main() -> int:
    ap = argparse.ArgumentParser(description="Khởi động bge_m3_server + vllm_qwen_server trên cùng GPU")
    ap.add_argument("--host", default="0.0.0.0", help="bind host cho cả 2 server (0.0.0.0 = mọi interface)")
    ap.add_argument("--embed-port", type=int, default=8000)
    ap.add_argument("--llm-port", type=int, default=8001)
    ap.add_argument("--llm-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-fraction", type=float, default=0.65,
                    help="VRAM cho vLLM (60-70%%); bge dùng ~10-20%% còn lại")
    ap.add_argument("--embed-batch-size", type=int, default=16)
    ap.add_argument("--embed-max-length", type=int, default=1024)
    ap.add_argument("--embed-timeout", type=float, default=300.0, help="giây chờ bge ready (load model + tải lần đầu)")
    ap.add_argument("--llm-timeout", type=float, default=900.0, help="giây chờ vLLM ready (load + tải lần đầu)")
    ap.add_argument("--no-llm", action="store_true", help="chỉ chạy embed server")
    ap.add_argument("--no-embed", action="store_true", help="chỉ chạy LLM server (bỏ qua thứ tự bge-trước)")
    args = ap.parse_args()

    if not EMBED_SCRIPT.exists() or not LLM_SCRIPT.exists():
        print(f"⚠️ Không thấy script: {EMBED_SCRIPT} / {LLM_SCRIPT}", file=sys.stderr)
        return 2

    embed: subprocess.Popen | None = None
    llm: subprocess.Popen | None = None

    # POSIX: start_new_session → mỗi con 1 process group riêng → kill được cả cây (vLLM grandchild).
    # Windows: CREATE_NEW_PROCESS_GROUP → taskkill /T kill cây.
    popen_kwargs: dict = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
        "env": os.environ,
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    else:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        # ---- 1) bge (embed) TRƯỚC ----
        if not args.no_embed:
            embed_cmd = [
                sys.executable, str(EMBED_SCRIPT),
                "--host", args.host, "--port", str(args.embed_port),
                "--batch-size", str(args.embed_batch_size),
                "--max-length", str(args.embed_max_length),
            ]
            print(f"[serve_all] ▶ bge_m3_server: {' '.join(embed_cmd[1:])}", flush=True)
            embed = subprocess.Popen(embed_cmd, **popen_kwargs)
            _pipe(embed, "[embed]")
            if not _wait_health(f"http://127.0.0.1:{args.embed_port}/health", "embed", args.embed_timeout):
                print("[serve_all] ❌ bge không lên — không start vLLM.", file=sys.stderr)
                _kill_group(embed)
                return 1

        # ---- 2) vLLM (LLM) SAU ----
        if not args.no_llm:
            llm_cmd = [
                sys.executable, str(LLM_SCRIPT),
                "--host", args.host, "--port", str(args.llm_port),
                "--model", args.llm_model,
                "--max-model-len", str(args.max_model_len),
                "--gpu-memory-fraction", str(args.gpu_memory_fraction),
            ]
            print(f"[serve_all] ▶ vllm_qwen_server: {' '.join(llm_cmd[1:])}", flush=True)
            llm = subprocess.Popen(llm_cmd, **popen_kwargs)
            _pipe(llm, "[llm]")
            if not _wait_health(f"http://127.0.0.1:{args.llm_port}/health", "llm", args.llm_timeout):
                print("[serve_all] ❌ vLLM không lên.", file=sys.stderr)
                _kill_group(llm)
                if embed is not None:
                    _kill_group(embed)
                return 1

        api_key = os.environ.get("VLLM_API_KEY", "vllm")
        _print_yaml(args.host, args.embed_port, args.llm_port, api_key, args.llm_model)
        print("[serve_all] ✅ Cả 2 service sẵn sàng. Ctrl-C để tắt.\n", flush=True)

        # ---- chờ đến khi 1 process chết hoặc Ctrl-C ----
        while True:
            if embed is not None and embed.poll() is not None:
                print(f"[serve_all] bge thoát (code {embed.returncode}) — tắt vLLM.", file=sys.stderr)
                break
            if llm is not None and llm.poll() is not None:
                print(f"[serve_all] vLLM thoát (code {llm.returncode}) — tắt bge.", file=sys.stderr)
                break
            time.sleep(2)
        return 0

    except KeyboardInterrupt:
        print("\n[serve_all] Ctrl-C — đang tắt cả 2 process group...", file=sys.stderr)
        return 0
    finally:
        if llm is not None:
            _kill_group(llm)
        if embed is not None:
            _kill_group(embed)
        print("[serve_all] đã tắt sạch.", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())