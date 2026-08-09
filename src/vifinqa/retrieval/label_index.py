"""label_index.py — lexical index chi_tieu → (report_id, table_id) từ evidence.

Tầng B của kiến trúc retrieval (item-level lexical recall):

    Câu hỏi "Lãi tiền gửi ... VJC 2018"
        │  metric_tokens = [lai, tien, gui]
        ▼
    label_index.lookup(...) ──► [(rid, table_id)] chứa label khớp
        │  filter theo ticker + year
        ▼
    union bảng vào evidence (bất kể top-k dense)

Tại sao cần: index dense hiện tại là 1 vector/1 bảng → "trung bình hoá" 40 chỉ tiêu
→ bảng dài bị pha loãng, miss chỉ tiêu ở dòng cuối (Q7 "Quỹ khen thưởng" dòng 11).
Nhưng `evidence/*.csv` đã có sẵn `chi_tieu` ASCII-chuẩn-hoá TỪNG DÒNG (không dilution),
và 99.5% câu hỏi khớp ≥2 token với chi_tieu (đã đo 200 câu đầu). → lexical index là
tầng CHÍNH cho exact-label lookup; dense chỉ là fallback cho câu phức/đa công ty.

Nguồn: evidence/{rid}__table_{N}.csv (đã rebuild, 146,246 file). Không cần embed.
Cache: pickle `data/derived/label_index.pkl` — build vài phút, reload ~giây.
"""

from __future__ import annotations

import pickle
import re
import unicodedata
from pathlib import Path

# Lấy từ deterministic (tránh import vòng) — normalize_label tương đương numbers.py
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_label(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.replace("Đ", "D").replace("đ", "d")
    return re.sub(r"\s+", " ", s.lower()).strip()


def _tokens(label: str) -> set[str]:
    """Token hoá chi_tieu: bỏ dấu + tách [a-z0-9]+, bỏ token quá ngắn (số mã...)."""
    return {t for t in _TOKEN_RE.findall(label) if len(t) >= 3}


class LabelIndex:
    """Inverted index: token → list[(rid, table_id, label)] + metadata để lookup."""

    def __init__(self, data: dict | None = None):
        # token → list of (rid, table_id) — label chứa token đó
        self._token_map: dict[str, list[tuple[str, str]]] = data or {}
        # (rid, table_id) → set(token) để tính overlap
        self._entry_tokens: dict[tuple[str, str], set[str]] = {}
        # (rid, table_id) → label tốt nhất (để trả về/giải thích)
        self._entry_label: dict[tuple[str, str], str] = {}

    @classmethod
    def build(cls, evidence_dir: Path, progress_every: int = 5000) -> "LabelIndex":
        """Scan mọi evidence CSV → index. Không cần pandas — đọc dòng thô."""
        idx = cls()
        token_map: dict[str, list[tuple[str, str]]] = {}
        entry_tokens: dict[tuple[str, str], set[str]] = {}
        entry_label: dict[tuple[str, str], str] = {}

        files = sorted(evidence_dir.glob("*.csv"))
        n_files = len(files)
        for i, f in enumerate(files, 1):
            # filename: {rid}__table_{N}.csv — tách theo marker __table_
            name = f.stem
            marker = "__table_"
            pos = name.find(marker)
            if pos < 0:
                continue
            rid = name[:pos]
            tid = "table_" + name[pos + len(marker):]
            key = (rid, tid)
            try:
                with open(f, encoding="utf-8", errors="replace") as fh:
                    header = fh.readline()
                    if "chi_tieu" not in header:
                        continue
                    for line in fh:
                        # cột 0 = chi_tieu (CSV quote-aware: bỏ " nếu có)
                        if not line.strip():
                            continue
                        if line.startswith('"'):
                            # chi_tieu có dấu phẩy → bị quote
                            end = line.find('"', 1)
                            label = line[1:end] if end > 0 else ""
                        else:
                            label = line.split(",", 1)[0]
                        label = label.strip()
                        if not label:
                            continue
                        norm = normalize_label(label)
                        toks = _tokens(norm)
                        if not toks:
                            continue
                        entry_tokens.setdefault(key, set()).update(toks)
                        # label ngắn nhất (đại diện) — tránh label dài lặp
                        cur = entry_label.get(key, "")
                        if not cur or len(norm) < len(cur):
                            entry_label[key] = norm
            except Exception:
                continue
            if i % progress_every == 0:
                print(f"  ...{i}/{n_files} files", flush=True)

        # đảo: token → entry list
        for key, toks in entry_tokens.items():
            for t in toks:
                token_map.setdefault(t, []).append(key)
        idx._token_map = token_map
        idx._entry_tokens = entry_tokens
        idx._entry_label = entry_label
        print(f"Build xong: {n_files} files → {len(token_map)} token / "
              f"{len(entry_tokens)} entry (rid,table)")
        return idx

    # ------------------------------------------------------------ lookup
    def lookup(
        self,
        metric_tokens: list[str],
        tickers: set[str] | None = None,
        years: set[int] | None = None,
        min_overlap: int = 2,
    ) -> list[tuple[str, str, float]]:
        """Trả [(rid, table_id, overlap_score)] có label chứa ≥min_overlap metric token.

        Filter theo ticker + year nếu có (tránh match nhầm sang công ty/năm khác cùng label).
        Score = số token khớp / tổng token metric (thưởng coverage).
        """
        mtoks = {t for t in metric_tokens if len(t) >= 3}
        if len(mtoks) < min_overlap:
            return []

        # gom candidate từ các token metric
        cand: dict[tuple[str, str], int] = {}
        for t in mtoks:
            for key in self._token_map.get(t, []):
                cand[key] = cand.get(key, 0) + 1

        # filter + score
        out: list[tuple[str, str, float]] = []
        for key, hits in cand.items():
            if hits < min_overlap:
                continue
            rid, tid = key
            if tickers:
                tkr = rid.split("_financial_statements_")[0] if "_financial_statements_" in rid else rid.split("_")[0]
                if tkr.upper() not in {t.upper() for t in tickers}:
                    continue
            if years:
                m = re.search(r"(?<!\d)(20\d\d)(?!\d)", rid)
                if m and int(m.group(0)) not in years:
                    continue
            # overlap/coverage score
            score = hits / len(mtoks)
            out.append((rid, tid, round(score, 3)))

        out.sort(key=lambda x: x[2], reverse=True)
        return out

    def label_of(self, rid: str, table_id: str) -> str:
        return self._entry_label.get((rid, table_id), "")

    # ------------------------------------------------------------ save/load
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "token_map": self._token_map,
            "entry_tokens": {f"{k[0]}|{k[1]}": sorted(v) for k, v in self._entry_tokens.items()},
            "entry_label": {f"{k[0]}|{k[1]}": v for k, v in self._entry_label.items()},
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=4)

    @classmethod
    def load(cls, path: Path) -> "LabelIndex":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        idx = cls()
        idx._token_map = payload["token_map"]
        idx._entry_tokens = {
            tuple(k.split("|")): set(v) for k, v in payload["entry_tokens"].items()
        }
        idx._entry_label = {
            tuple(k.split("|")): v for k, v in payload["entry_label"].items()
        }
        return idx


def metric_tokens_from(
    question: str,
    stopwords: set[str] | None = None,
    tickers: set[str] | None = None,
) -> list[str]:
    """Rút token chỉ tiêu từ câu hỏi (bỏ entity/năm/đơn vị/filler).

    `tickers` (vd {'VJC'}) → bỏ token trùng ticker. Bỏ token toàn số (năm/kỳ).
    """
    default_stop = {
        "cua", "cong", "ty", "cty", "ngan", "hang", "tmcp", "co", "phan", "tap",
        "doan", "ctcp", "nam", "la", "bao", "nhieu", "trieu", "ty", "dong", "nghin",
        "va", "cac", "khoan", "theo", "den", "ngay", "cuoi", "dau", "trong", "vong",
        "da", "tai", "cho", "voi", "tu", "thuoc", "nhom", "tong", "so", "du",
        "phai", "tra", "thu", "khac", "hinh", "vao", "nguoi", "ban", "khong",
        "lon", "nho", "khi", "neu", "thi", "hoac", "duoc", "doi", "giua", "den",
        "thang", "nam", "loai", "dvt", "vnd", "tri", "co", "la", "ma",
    }
    stop = default_stop if stopwords is None else stopwords
    # bỏ token trùng ticker (thường viết hoa trong ngoặc)
    if tickers:
        stop = stop | {t.lower() for t in tickers}
    toks = _TOKEN_RE.findall(normalize_label(question))
    return [t for t in toks if t not in stop and len(t) >= 3 and not t.isdigit()]
