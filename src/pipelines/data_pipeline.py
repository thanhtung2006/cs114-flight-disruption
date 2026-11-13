"""Pipeline bước chung: nạp dữ liệu BTS theo tháng -> tạo nhãn -> gán split -> kiểm tra.

Mọi hằng số (usecols, dtype, ngưỡng trễ, mốc chia) đọc từ config/config.yaml.
Không tạo đặc trưng ở đây.
"""
from __future__ import annotations

import gc
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipelines.utils import get_path, peak_rss_mb, rss_mb

LABEL_STATUSES = ["negative", "positive", "undetermined"]


# --------------------------------------------------------------------------- #
# Nạp dữ liệu
# --------------------------------------------------------------------------- #
def expected_months(cfg: dict) -> list[tuple[int, int]]:
    """Danh sách (năm, tháng) cần có, theo data.raw_months trong config."""
    rng = cfg["data"]["raw_months"]
    periods = pd.period_range(rng["start"], rng["end"], freq="M")
    return [(p.year, p.month) for p in periods]


def list_raw_files(cfg: dict, raw_dir: Path | None = None) -> dict[tuple[int, int], Path]:
    """Quét thư mục raw, ghép file với từng tháng. Thiếu hoặc trùng tháng thì raise."""
    raw_dir = Path(raw_dir) if raw_dir is not None else get_path(cfg["data"]["raw_dir"])
    pattern = re.compile(cfg["data"]["raw_name_regex"])
    found: dict[tuple[int, int], list[Path]] = {}
    for f in sorted(raw_dir.iterdir()):
        m = pattern.search(f.name)
        if m and f.is_file():
            found.setdefault((int(m.group(1)), int(m.group(2))), []).append(f)

    expected = expected_months(cfg)
    missing = [ym for ym in expected if ym not in found]
    duplicated = {ym: fs for ym, fs in found.items() if len(fs) > 1}
    extra = sorted(ym for ym in found if ym not in expected)
    if extra:
        print(f"[CẢNH BÁO] Có file ngoài dải tháng trong config, sẽ bỏ qua: {extra}")
    if missing or duplicated:
        raise FileNotFoundError(
            f"Thiếu tháng: {[f'{y}-{m:02d}' for y, m in missing]}; "
            f"tháng có nhiều file: {[f'{y}-{m:02d}' for y, m in duplicated]}"
        )
    return {ym: found[ym][0] for ym in expected}


def _open_csv(path: Path):
    """Trả về (file object, tên csv). Với zip thì mở đúng file .csv bên trong."""
    if path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(path)
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(csvs) != 1:
            raise ValueError(f"{path.name}: cần đúng 1 file csv trong zip, thấy {csvs}")
        return zf.open(csvs[0]), csvs[0]
    return open(path, "rb"), path.name


def read_header(path: Path) -> list[str]:
    """Đọc dòng tiêu đề của file (để kiểm tra cột, phát hiện Unnamed: 109)."""
    fh, _ = _open_csv(path)
    with fh:
        return list(pd.read_csv(fh, nrows=0).columns)


def read_month(path: Path, cfg: dict, year: int, month: int) -> pd.DataFrame:
    """Đọc một tháng với usecols + dtype nhỏ; kiểm tra cột và tháng của FlightDate."""
    dcfg = cfg["data"]
    usecols = list(dcfg["usecols"])
    date_col = dcfg["date_col"]
    check_no_forbidden(usecols, cfg)

    header = read_header(path)
    missing_cols = [c for c in usecols if c not in header]
    if missing_cols:
        raise ValueError(f"{path.name}: thiếu cột {missing_cols}")

    fh, _ = _open_csv(path)
    with fh:
        df = pd.read_csv(
            fh,
            usecols=usecols,
            dtype=dcfg["dtypes"],
            parse_dates=[date_col],
            date_format="%Y-%m-%d",
        )
    # Bỏ cột rỗng (thực ra usecols đã loại, giữ để chắc chắn)
    df = df.drop(columns=[c for c in dcfg["drop_cols"] if c in df.columns])
    assert not any(c in df.columns for c in dcfg["drop_cols"])
    df = df[usecols]

    # Cờ hủy/chuyển hướng: phải là 0/1, không thiếu, rồi ép int8
    for c in dcfg["flag_cols"]:
        bad = ~df[c].isin([0.0, 1.0])
        if bad.any():
            raise ValueError(f"{path.name}: cột {c} có {int(bad.sum())} giá trị ngoài {{0,1}} hoặc thiếu")
        df[c] = df[c].astype("int8")

    # Mọi FlightDate phải thuộc đúng tháng của file
    dates = df[date_col]
    if dates.isna().any():
        raise ValueError(f"{path.name}: {int(dates.isna().sum())} dòng thiếu {date_col}")
    wrong = (dates.dt.year != year) | (dates.dt.month != month)
    if wrong.any():
        raise ValueError(f"{path.name}: {int(wrong.sum())} dòng có {date_col} ngoài tháng {year}-{month:02d}")
    return df


def check_no_forbidden(cols: list[str], cfg: dict) -> None:
    """Assert không có cột cấm (rò rỉ) trong danh sách cột."""
    forbidden = set(cfg["forbidden_cols"])
    prefixes = tuple(cfg.get("forbidden_prefixes", []))
    label_only = set(cfg.get("label_only_cols", []))  # vd Diverted bắt đầu bằng "Div" nhưng là cột nhãn
    leaked = [
        c for c in cols
        if c in forbidden or (prefixes and c.startswith(prefixes) and c not in label_only)
    ]
    assert not leaked, f"Cột cấm lọt vào dữ liệu: {leaked}"


def hhmm_stats(df: pd.DataFrame) -> dict:
    """Thống kê giờ dự kiến hhmm: số giá trị 2400 và số giá trị không hợp lệ (chỉ đếm, không sửa)."""
    out = {}
    for c in ["CRSDepTime", "CRSArrTime"]:
        v = df[c]
        invalid = (v < 0) | (v > 2400) | ((v % 100) >= 60)
        out[f"{c}_2400"] = int((v == 2400).sum())
        out[f"{c}_invalid"] = int(invalid.sum())
    return out


def duplicate_stats(df: pd.DataFrame, key: list[str]) -> dict:
    """Đếm dòng trùng toàn phần và trùng theo khóa chuyến bay (không xóa)."""
    return {
        "dup_full_rows": int(df.duplicated().sum()),
        "dup_key_rows": int(df.duplicated(subset=key).sum()),
    }


def _unify_categories(frames: list[pd.DataFrame]) -> None:
    """Đặt cùng một bộ category (đã sắp xếp) cho mọi tháng để concat vẫn giữ kiểu category."""
    cat_cols = [c for c in frames[0].columns if isinstance(frames[0][c].dtype, pd.CategoricalDtype)]
    for c in cat_cols:
        cats = sorted(set().union(*(set(f[c].cat.categories) for f in frames)))
        for f in frames:
            f[c] = f[c].cat.set_categories(cats)


def load_all_months(cfg: dict, raw_dir: Path | None = None, verbose: bool = True):
    """Nạp lần lượt từng tháng, in shape/RAM, rồi nối lại.

    Trả về (df, month_report) với month_report là DataFrame thống kê theo tháng.
    """
    files = list_raw_files(cfg, raw_dir)
    key = cfg["data"]["dup_key"]
    frames, rows = [], []
    for (year, month), path in files.items():
        t0 = time.perf_counter()
        header = read_header(path)
        df_m = read_month(path, cfg, year, month)
        rec = {
            "month": f"{year}-{month:02d}",
            "file": path.name,
            "n_header_cols": len(header),
            "has_unnamed_109": "Unnamed: 109" in header,
            "rows": len(df_m),
            "cols": df_m.shape[1],
            **hhmm_stats(df_m),
            **duplicate_stats(df_m, key),
            "seconds": round(time.perf_counter() - t0, 1),
            "rss_mb": round(rss_mb(), 0),
            "peak_rss_mb": round(peak_rss_mb(), 0),
        }
        rows.append(rec)
        if verbose:
            print(
                f"{rec['month']}: shape=({rec['rows']:,}, {rec['cols']}) | "
                f"2400 dep/arr={rec['CRSDepTime_2400']}/{rec['CRSArrTime_2400']} | "
                f"trùng toàn phần={rec['dup_full_rows']}, trùng khóa={rec['dup_key_rows']} | "
                f"{rec['seconds']}s | RSS={rec['rss_mb']:.0f} MB, đỉnh={rec['peak_rss_mb']:.0f} MB"
            )
        frames.append(df_m)
        del df_m
        gc.collect()

    _unify_categories(frames)
    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    report = pd.DataFrame(rows)
    assert len(report) == len(expected_months(cfg)), "Số tháng nạp không khớp config"
    assert len(df) == report["rows"].sum()
    return df, report


# --------------------------------------------------------------------------- #
# Nhãn
# --------------------------------------------------------------------------- #
def make_labels(df: pd.DataFrame, threshold: float, copy: bool = True) -> pd.DataFrame:
    """Tạo nhãn theo 3 trường hợp (không điền ArrDelay thiếu bằng 0).

    - positive: Cancelled == 1 hoặc Diverted == 1 hoặc ArrDelay >= threshold
    - negative: Cancelled == 0, Diverted == 0, ArrDelay hợp lệ và < threshold
    - undetermined: còn lại (y = NA)
    """
    out = df.copy() if copy else df
    arr = out["ArrDelay"].astype("float64")
    canc = pd.Series(out["Cancelled"], copy=False)
    div = pd.Series(out["Diverted"], copy=False)

    def _bool(s) -> np.ndarray:  # NA -> False
        return pd.Series(s).fillna(False).astype(bool).to_numpy()

    is_c = _bool(canc.eq(1))
    is_d = _bool(div.eq(1))
    late = (arr >= threshold).to_numpy()  # NaN -> False
    pos = is_c | is_d | late
    neg = _bool(canc.eq(0)) & _bool(div.eq(0)) & arr.notna().to_numpy() & (arr < threshold).to_numpy()
    assert not (pos & neg).any(), "Một dòng vừa positive vừa negative"

    status = np.where(pos, "positive", np.where(neg, "negative", "undetermined"))
    y = pd.array(np.where(pos, 1, 0), dtype="Int8")
    y[~(pos | neg)] = pd.NA

    out["is_cancelled"] = is_c
    out["is_diverted"] = is_d
    out["is_delay60"] = late & ~is_c & ~is_d
    out["label_status"] = pd.Categorical(status, categories=LABEL_STATUSES)
    out["y"] = y
    return out


def arrdelay_missing_check(df: pd.DataFrame, date_col: str = "FlightDate") -> pd.DataFrame:
    """Theo tháng: số ArrDelay thiếu so với hủy + chuyển hướng."""
    month = df[date_col].dt.to_period("M").astype(str)
    g = pd.DataFrame(
        {
            "month": month,
            "arr_missing": df["ArrDelay"].isna(),
            "cancelled": df["is_cancelled"],
            "diverted": df["is_diverted"],
            "missing_not_cd": df["ArrDelay"].isna() & ~df["is_cancelled"] & ~df["is_diverted"],
            "cd_with_arrdelay": df["ArrDelay"].notna() & (df["is_cancelled"] | df["is_diverted"]),
        }
    ).groupby("month").sum()
    g["cancel_plus_divert"] = g["cancelled"] + g["diverted"]
    g["diff"] = g["arr_missing"] - g["cancel_plus_divert"]
    return g.astype("int64")


# --------------------------------------------------------------------------- #
# Chia tập theo thời gian
# --------------------------------------------------------------------------- #
def _ranges(split_cfg: dict) -> list[dict]:
    return [
        {"name": r["name"], "start": pd.Timestamp(r["start"]), "end": pd.Timestamp(r["end"])}
        for r in split_cfg["ranges"]
    ]


def split_names(split_cfg: dict) -> list[str]:
    """Tên các split theo thứ tự xuất hiện đầu tiên (warmup, train, gap, val, test)."""
    return list(dict.fromkeys(r["name"] for r in split_cfg["ranges"]))


def validate_split_config(split_cfg: dict) -> None:
    """Các đoạn trong config: start <= end, tăng dần, không chồng nhau."""
    rs = _ranges(split_cfg)
    for r in rs:
        assert r["start"] <= r["end"], f"Đoạn {r['name']}: start > end"
    for a, b in zip(rs, rs[1:]):
        assert a["end"] < b["start"], f"Đoạn {a['name']} ({a['end'].date()}) chồng/sai thứ tự với {b['name']} ({b['start'].date()})"


def segment_index(dates: pd.Series, split_cfg: dict) -> np.ndarray:
    """Chỉ số đoạn (theo thứ tự trong config) của từng ngày; -1 nếu không thuộc đoạn nào."""
    validate_split_config(split_cfg)
    seg = np.full(len(dates), -1, dtype=np.int8)
    d = pd.to_datetime(dates).to_numpy()
    for i, r in enumerate(_ranges(split_cfg)):
        mask = (d >= r["start"].to_datetime64()) & (d <= r["end"].to_datetime64())
        assert not (seg[mask] >= 0).any(), "Một ngày thuộc hai đoạn"
        seg[mask] = i
    return seg


def assign_split(df: pd.DataFrame, split_cfg: dict, date_col: str = "FlightDate", copy: bool = True) -> pd.DataFrame:
    """Gán cột split theo mốc ngày trong config. Ngày ngoài mọi đoạn -> lỗi."""
    out = df.copy() if copy else df
    seg = segment_index(out[date_col], split_cfg)
    if (seg < 0).any():
        bad = pd.to_datetime(out.loc[seg < 0, date_col])
        raise ValueError(f"{int((seg < 0).sum())} dòng không thuộc đoạn nào (vd {bad.min().date()} .. {bad.max().date()})")
    names = np.array([r["name"] for r in split_cfg["ranges"]])
    out[split_cfg["col"]] = pd.Categorical(names[seg], categories=split_names(split_cfg))
    return out


def validate_split(df: pd.DataFrame, split_cfg: dict, date_col: str = "FlightDate") -> pd.DataFrame:
    """Kiểm tra bắt buộc sau khi chia (assert). Trả về bảng khoảng ngày theo từng đoạn."""
    validate_split_config(split_cfg)
    col = split_cfg["col"]
    dates = pd.to_datetime(df[date_col])
    seg = segment_index(dates, split_cfg)
    assert (seg >= 0).all(), "Có dòng không thuộc đoạn nào"
    names = np.array([r["name"] for r in split_cfg["ranges"]])
    assert (df[col].astype(str).to_numpy() == names[seg]).all(), "Cột split không khớp mốc ngày trong config"

    # Theo từng đoạn: khoảng ngày thực tế, thứ tự thời gian
    segs = []
    for i, r in enumerate(split_cfg["ranges"]):
        d = dates[seg == i]
        segs.append({"segment": i, "split": r["name"], "cfg_start": r["start"], "cfg_end": r["end"],
                     "rows": len(d), "date_min": d.min() if len(d) else pd.NaT,
                     "date_max": d.max() if len(d) else pd.NaT})
    seg_df = pd.DataFrame(segs)
    nonempty = seg_df[seg_df["rows"] > 0]
    for (_, a), (_, b) in zip(nonempty.iterrows(), nonempty.iloc[1:].iterrows()):
        assert a["date_max"] < b["date_min"], f"Đoạn {a['split']} không kết thúc trước {b['split']}"

    # Các split không giao nhau về ngày
    day_sets = {n: set(dates[df[col].astype(str) == n].dt.normalize().unique()) for n in split_names(split_cfg)}
    keys = list(day_sets)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert not (day_sets[a] & day_sets[b]), f"Split {a} và {b} có chung ngày"
    return seg_df


def split_summary(df: pd.DataFrame, split_cfg: dict, date_col: str = "FlightDate") -> pd.DataFrame:
    """Bảng theo split: số dòng, khoảng ngày, số chưa xác định, tỉ lệ y=1 và từng thành phần.

    Tỉ lệ tính trên các dòng có nhãn xác định.
    """
    col = split_cfg["col"]
    rows = []
    for name in split_names(split_cfg) + ["ALL"]:
        part = df if name == "ALL" else df[df[col] == name]
        det = part["label_status"] != "undetermined"
        n_det = int(det.sum())
        rows.append({
            "split": name,
            "rows": len(part),
            "date_min": part[date_col].min().date(),
            "date_max": part[date_col].max().date(),
            "n_days": part[date_col].nunique(),
            "undetermined": int((~det).sum()),
            "determined": n_det,
            "y1": int(part["y"].sum()),
            "y1_rate": float(part["y"].mean()) if n_det else float("nan"),
            "cancel_rate": float(part.loc[det, "is_cancelled"].mean()) if n_det else float("nan"),
            "divert_rate": float(part.loc[det, "is_diverted"].mean()) if n_det else float("nan"),
            "delay60_rate": float(part.loc[det, "is_delay60"].mean()) if n_det else float("nan"),
        })
    return pd.DataFrame(rows)
