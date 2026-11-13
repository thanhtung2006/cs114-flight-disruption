"""Chọn đặc trưng và tách dòng theo split cho huấn luyện / đánh giá.

schedule_v0: chỉ dùng cột lịch bay có sẵn trong flights.parquet, không biến đổi thêm
(mô hình cây không cần chuẩn hóa; hhmm tăng cùng chiều với thời gian trong ngày).
Đặc trưng lịch sử 7/30 ngày làm ở giai đoạn sau.
"""
from __future__ import annotations

import pandas as pd

from src.pipelines.data_pipeline import check_no_forbidden

COMPONENT_COLS = ["is_cancelled", "is_diverted", "is_delay60"]


def feature_columns(cfg: dict, feature_set: str) -> tuple[list[str], list[str]]:
    """Trả về (tất cả cột đặc trưng, cột phân loại) của một bộ đặc trưng trong config.

    Assert không có cột cấm, cột chỉ dùng tạo nhãn, hay cột nhãn/split/ngày.
    """
    fs = cfg["features"][feature_set]
    categorical = list(fs.get("categorical", []))
    numeric = list(fs.get("numeric", []))
    cols = categorical + numeric
    assert len(cols) == len(set(cols)), f"Cột đặc trưng bị lặp: {cols}"

    check_no_forbidden(cols, cfg)
    banned = set(cfg.get("label_only_cols", [])) | set(cfg["features"].get("never_features", []))
    leaked = [c for c in cols if c in banned]
    assert not leaked, f"Cột không được làm đầu vào: {leaked}"
    return cols, categorical


def select_split(df: pd.DataFrame, cfg: dict, split_name: str, feature_set: str):
    """Lấy các dòng của một split có nhãn xác định.

    Trả về (X, y, components): X chỉ gồm cột đặc trưng, y kiểu int8,
    components là 3 cờ thành phần (để báo cáo riêng từng loại gián đoạn).
    """
    cols, categorical = feature_columns(cfg, feature_set)
    split_col = cfg["split"]["col"]
    mask = (df[split_col] == split_name) & (df["label_status"] != "undetermined")
    part = df.loc[mask]
    assert part[cfg["target"]].notna().all(), "Còn nhãn NA sau khi loại undetermined"

    X = part[cols].copy()
    for c in categorical:
        if not isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].astype("category")
    y = part[cfg["target"]].astype("int8")
    components = part[COMPONENT_COLS].astype(bool)
    return X, y, components
