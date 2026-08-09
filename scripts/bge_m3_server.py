"""bge_m3_server.py — 1 file chạy trên GPU thuê, export API /embed cho local.

Cách dùng (GPU thuê — Vast/RunPod, template PyTorch, RTX 3060 12GB):
    python bge_m3_server.py                    # tự cài deps + load model + serve
    python bge_m3_server.py --batch-size 16 --max-length 1024

API:
    POST /embed      {"texts": [...], "return_dense": true} → {"dense": [[...]]}
    POST /embed_bin  cùng request + "binary": true → raw float32 bytes (nhanh hơn 5x)
    GET  /health     → {"status": "ok"}

Local (máy của bạn) chỉ cần đổi configs/api.yaml:
    retrieval.embedding.base_url: http://<GPU_IP>:8000
rồi chạy:  python scripts/build_retrieval_index.py

Tự cài deps (gồm torch), tự load model BAAI/bge-m3 fp16, tự test embed dim=1024 khi
khởi động. Chống OOM: nếu encode batch lớn lỗi → tự tách từng chunk 8 texts.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

# ---------------------------------------------------------------- auto-install
# FlagEmbedding 1.4.0 kéo peft/deepspeed/transformers-mới → lỗi import (torchvision.io,
# BloomPreTrainedModel, TrainingArguments...). Pin bản ổn định 1.2.10 + transformers 4.44.
# torch phải khớp CUDA driver của GPU thuê — cài sai bản → GPU không dùng được.
_REQUIRED = {"torch", "torchvision", "fastapi", "uvicorn", "FlagEmbedding", "pydantic"}
_PINNED = {
    "FlagEmbedding": "FlagEmbedding==1.2.10",
    "transformers": "transformers==4.44.2",
    "sentence_transformers": "sentence-transformers==2.7.0",
}


def _pip_install(pkgs: list[str]) -> None:
    import subprocess as _sp

    _sp.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)


def _detect_cuda_driver() -> str | None:
    """nvidia-smi → '12.4' | None. Dùng để chọn index torch tương ứng."""
    import re
    import subprocess as _sp

    try:
        out = _sp.run(
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


def _install_torch() -> None:
    """Cài bộ torch+torchvision khớp driver CUDA (cu124 cho driver >=12.4).

    Template PyTorch thường có SẴN torch bản mới (cu130) nhưng driver chỉ 12.4 → GPU
    không dùng được. Phải UNINSTALL rồi cài lại cùng bản torch+torchvision từ index cu124.
    """
    import subprocess as _sp

    driver = _detect_cuda_driver()
    index = None
    torch_ver = "2.4.1"
    tv_ver = "0.19.1"
    if driver:
        major, minor = (int(x) for x in driver.split("."))
        if (major, minor) >= (12, 4):
            index = "https://download.pytorch.org/whl/cu124"
        elif (major, minor) >= (12, 1):
            index = "https://download.pytorch.org/whl/cu121"
            torch_ver, tv_ver = "2.4.1", "0.19.1"
    if not index:
        # driver cũ / không thấy → torch CPU
        _sp.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision"])
    else:
        print(f"[deps] gỡ torch cũ (torch {_torch_ver()} / {_torchvision_ver()}), "
              f"cài torch=={torch_ver} torchvision=={tv_ver} từ {index}")
        _sp.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision"])
        _sp.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            f"torch=={torch_ver}", f"torchvision=={tv_ver}",
            "--index-url", index,
        ])
    # in thông tin để chẩn đoán
    try:
        import torch

        print(f"[deps] torch {torch.__version__} | CUDA build={torch.version.cuda} "
              f"| available={torch.cuda.is_available()}")
    except Exception:
        pass
    if not _torch_cuda_ok():
        print("[deps] ⚠️ CUDA vẫn không khả dụng — kiểm tra: nvidia-smi có GPU? "
              "container có mount /dev/nvidia*? driver đủ mới?")


def _torch_ver() -> str:
    try:
        import torch

        return torch.__version__
    except Exception:
        return "?"


def _torchvision_ver() -> str:
    try:
        import torchvision

        return torchvision.__version__
    except Exception:
        return "?"


def _ensure_deps() -> None:
    missing = []
    for pkg in _REQUIRED:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    # torch có sẵn nhưng không khớp driver → cài lại
    if "torch" not in missing and not _torch_cuda_ok():
        print("[deps] torch không dùng được CUDA — cài lại khớp driver")
        missing.append("torch")
    if "torch" in missing:
        _install_torch()
    missing = [m for m in missing if m != "torch"]
    if missing:
        print(f"[deps] cài thiếu: {missing}")
        _pip_install(missing)
    # Pin version ổn định (tránh FlagEmbedding 1.4 + transformers mới gây lỗi import)
    import importlib.metadata as _md

    def _ver(pkg: str) -> str | None:
        try:
            return _md.version(pkg)
        except Exception:
            return None

    for pkg, spec in _PINNED.items():
        want = spec.split("==")[1]
        if _ver(pkg) != want:
            print(f"[deps] pin {pkg} {_ver(pkg)} → {want}")
            _pip_install([spec])


# ---------------------------------------------------------------- model
_model = None


def _gpu_info() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "CPU (không có CUDA!)"
        return f"CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)} | " \
               f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
    except Exception:
        return "chưa rõ"


def get_model():
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel

        t0 = time.time()
        print(f"[model] GPU: {_gpu_info()}")
        last = None
        for attempt in range(3):  # retry download model (HF hay đứt kết nối từ VN)
            try:
                _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
                break
            except Exception as e:  # noqa: BLE001
                last = e
                print(f"[model] load lỗi (attempt {attempt + 1}): {type(e).__name__} {str(e)[:120]}")
                if attempt == 2:
                    raise
        print(f"[model] BAAI/bge-m3 loaded in {time.time() - t0:.0f}s")
    return _model


# ---------------------------------------------------------------- app
def make_app(batch_size: int, max_length: int, device: str = "auto"):
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="BGE-M3 Embedding API", version="0.1.0")

    class EmbedRequest(BaseModel):
        texts: list[str]
        return_dense: bool = True
        return_sparse: bool = False
        return_colbert: bool = False
        binary: bool = False   # True → trả raw float32 bytes (giảm 5x so với JSON)

    class EmbedResponse(BaseModel):
        dense: list[list[float]] | None = None
        sparse: list[dict] | None = None
        colbert: list[list[list[float]]] | None = None

    @app.on_event("startup")
    def _startup() -> None:
        model = get_model()
        out = model.encode(["test embedding ok"], batch_size=1, max_length=64,
                           return_dense=True, return_sparse=False, return_colbert=False)
        dim = out["dense_vecs"].shape[-1]
        print(f"[startup] test embed OK — dim={dim}")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    def _encode(req: EmbedRequest) -> dict:
        """Encode texts → dict {'dense_vecs': ndarray, ...}.

        Chống OOM: tự tách chunk nhỏ nếu batch lớn lỗi.
        Tương thích FlagEmbedding 1.2.x (`return_colbert`) và 1.4.x (`return_colbert_vecs`).
        """
        import inspect as _inspect

        encode_sig = _inspect.signature(model.encode)
        colbert_kw = "return_colbert" if "return_colbert" in encode_sig.parameters else "return_colbert_vecs"

        def _do(texts, bs):
            kwargs = {
                "return_dense": req.return_dense,
                "return_sparse": req.return_sparse,
                colbert_kw: req.return_colbert,
            }
            return model.encode(texts, batch_size=bs, max_length=max_length, **kwargs)

        n = len(req.texts)
        b = min(batch_size, n or 1)
        try:
            return _do(req.texts, b)
        except Exception:
            # OOM hoặc lỗi → encode từng chunk nhỏ (an toàn GPU yếu)
            dense_chunks, sparse_chunks, colbert_chunks = [], [], []
            step = 8
            for i in range(0, n, step):
                chunk = req.texts[i:i + step]
                partial = _do(chunk, len(chunk))
                if req.return_dense:
                    dense_chunks.append(partial["dense_vecs"])
                if req.return_sparse:
                    sparse_chunks.extend(partial.get("lexical_weights", []))
                if req.return_colbert:
                    colbert_chunks.extend(partial.get("colbert_vecs", []))
            import numpy as np

            out = {"dense_vecs": np.concatenate(dense_chunks) if dense_chunks else None}
            if req.return_sparse:
                out["lexical_weights"] = sparse_chunks
            if req.return_colbert:
                out["colbert_vecs"] = colbert_chunks
            return out

    @app.post("/embed", response_model=EmbedResponse)
    def embed(req: EmbedRequest) -> EmbedResponse:
        model = get_model()
        t0 = time.time()
        out = _encode(req)
        resp = EmbedResponse()
        if req.return_dense and out.get("dense_vecs") is not None:
            resp.dense = out["dense_vecs"].tolist()
        if req.return_sparse:
            resp.sparse = [{str(k): float(v) for k, v in lw.items()}
                           for lw in out.get("lexical_weights", [])]
        if req.return_colbert:
            resp.colbert = [cv.tolist() for cv in out.get("colbert_vecs", [])]
        print(f"[embed] n={len(req.texts)} | {time.time()-t0:.1f}s | "
              f"{len(req.texts)/max(time.time()-t0,1e-6):.0f} texts/s", flush=True)
        return resp

    @app.post("/embed_bin")
    def embed_bin(req: EmbedRequest):
        """Trả raw float32 bytes (N, dim) — nhanh hơn JSON ~5x. Client dùng np.frombuffer."""
        from fastapi.responses import Response

        model = get_model()
        t0 = time.time()
        out = _encode(req)
        if not req.return_dense or out.get("dense_vecs") is None:
            return Response(content=b"", media_type="application/octet-stream")
        import numpy as np

        dense = np.asarray(out["dense_vecs"], dtype=np.float32)
        print(f"[embed_bin] n={len(req.texts)} | {time.time()-t0:.1f}s | "
              f"{len(req.texts)/max(time.time()-t0,1e-6):.0f} texts/s", flush=True)
        return Response(content=dense.tobytes(), media_type="application/octet-stream")

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description="BGE-M3 embed API — chạy trên GPU thuê")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=16, help="GPU batch (RTX 3060 12GB fp16: 16)")
    ap.add_argument("--max-length", type=int, default=1024, help="token cap (chunk 2000 chars ≈ 700 token)")
    ap.add_argument("--workers", type=int, default=1, help="uvicorn workers (giữ 1 — model không thread-safe)")
    args = ap.parse_args()

    _ensure_deps()
    app = make_app(args.batch_size, args.max_length)

    import uvicorn

    print(f"[serve] http://{args.host}:{args.port} | batch={args.batch_size} "
          f"| max_length={args.max_length} | workers={args.workers}")
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
