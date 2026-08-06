"""pack.py — đóng gói submission.zip (submission.json + data/** ở root).

Quy ước §3.2: submission.json và data/ nằm trực tiếp ở cấp ngoài cùng — không bọc
thư mục cha; đúng 1 file .json. Sort entries cho deterministic.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path


def pack(out_dir: Path) -> Path:
    """Tạo submission.zip tại out_dir. Trả path ZIP."""
    out_dir = Path(out_dir)
    submission_json = out_dir / "submission.json"
    data_dir = out_dir / "data"
    if not submission_json.exists():
        raise FileNotFoundError(submission_json)

    zip_path = out_dir / "submission.zip"
    entries: list[tuple[str, Path]] = []
    entries.append(("submission.json", submission_json))
    if data_dir.exists():
        for p in sorted(data_dir.rglob("*")):
            if p.is_file():
                rel = p.relative_to(out_dir).as_posix()
                entries.append((rel, p))

    # check đúng 1 file .json ở root
    json_at_root = [name for name, _ in entries if "/" not in name and name.endswith(".json")]
    if len(json_at_root) != 1:
        raise ValueError(f"phải có đúng 1 file .json ở root, thấy {len(json_at_root)}")

    manifest_lines = []
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, src in entries:
            zf.write(src, arcname=name)
            manifest_lines.append(f"{name}\t{src.stat().st_size}")

    # checksum.txt (ngoài ZIP cho audit; không bắt buộc trong ZIP)
    h = hashlib.sha256("\n".join(manifest_lines).encode("utf-8")).hexdigest()
    (out_dir / "checksum.txt").write_text(
        "\n".join(manifest_lines) + f"\nsha256\t{h}\n", encoding="utf-8"
    )
    return zip_path