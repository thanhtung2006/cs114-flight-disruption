"""Test quy tắc nhãn 3 trường hợp (dữ liệu nhỏ tự tạo)."""
import numpy as np
import pandas as pd
import pytest

from src.pipelines.data_pipeline import make_labels
from src.pipelines.utils import load_config

THR = load_config()["label"]["delay_threshold_min"]


def _df(rows):
    return pd.DataFrame(rows, columns=["Cancelled", "Diverted", "ArrDelay"]).astype(
        {"Cancelled": "int8", "Diverted": "int8", "ArrDelay": "float32"}
    )


@pytest.fixture
def labeled():
    rows = [
        (1, 0, np.nan),   # 0: hủy -> positive
        (0, 1, np.nan),   # 1: chuyển hướng -> positive
        (0, 0, 60.0),     # 2: trễ đúng ngưỡng -> positive
        (0, 0, 59.9),     # 3: dưới ngưỡng -> negative
        (0, 0, -15.0),    # 4: đến sớm -> negative
        (0, 0, np.nan),   # 5: thiếu ArrDelay, không hủy/chuyển hướng -> undetermined
        (0, 0, 300.0),    # 6: trễ nhiều -> positive
        (1, 0, 120.0),    # 7: hủy (dù có ArrDelay) -> positive, không tính is_delay60
        (0, 1, 75.0),     # 8: chuyển hướng + ArrDelay >= 60 -> positive, không tính is_delay60
        (0, 0, 0.0),      # 9: đúng giờ -> negative
    ]
    return make_labels(_df(rows), THR)


def test_threshold_is_60():
    assert THR == 60


def test_label_status(labeled):
    expected = ["positive", "positive", "positive", "negative", "negative",
                "undetermined", "positive", "positive", "positive", "negative"]
    assert labeled["label_status"].astype(str).tolist() == expected


def test_y_values_and_dtype(labeled):
    assert labeled["y"].dtype == "Int8"
    y = labeled["y"]
    assert y.isna().tolist() == [False] * 5 + [True] + [False] * 4
    assert y.dropna().astype(int).tolist() == [1, 1, 1, 0, 0, 1, 1, 1, 0]


def test_missing_arrdelay_not_filled_with_zero(labeled):
    # ArrDelay thiếu mà không hủy/chuyển hướng: không được thành nhãn 0
    row = labeled.iloc[5]
    assert row["label_status"] == "undetermined"
    assert pd.isna(row["y"])
    assert pd.isna(row["ArrDelay"])


def test_components(labeled):
    assert labeled["is_cancelled"].tolist() == [True, False, False, False, False, False, False, True, False, False]
    assert labeled["is_diverted"].tolist() == [False, True, False, False, False, False, False, False, True, False]
    assert labeled["is_delay60"].tolist() == [False, False, True, False, False, False, True, False, False, False]


def test_components_consistent_with_y(labeled):
    pos = labeled["label_status"] == "positive"
    any_comp = labeled["is_cancelled"] | labeled["is_diverted"] | labeled["is_delay60"]
    assert (pos == any_comp).all()


def test_nullable_flags_go_undetermined():
    # Cờ hủy/chuyển hướng thiếu (NA) và ArrDelay < 60 -> không đủ để kết luận negative
    df = pd.DataFrame({
        "Cancelled": pd.array([pd.NA, 0, pd.NA], dtype="Int8"),
        "Diverted": pd.array([0, pd.NA, 0], dtype="Int8"),
        "ArrDelay": pd.array([10.0, 10.0, 90.0], dtype="float32"),
    })
    out = make_labels(df, THR)
    assert out["label_status"].astype(str).tolist() == ["undetermined", "undetermined", "positive"]


def test_forbidden_columns_check():
    from src.pipelines.data_pipeline import check_no_forbidden

    cfg = load_config()
    check_no_forbidden(cfg["data"]["usecols"], cfg)  # usecols hợp lệ (có Diverted)
    for leaked in (["DepDelay"], ["DivArrDelay"], ["Div1Airport"], ["WeatherDelay"]):
        with pytest.raises(AssertionError):
            check_no_forbidden(cfg["data"]["usecols"] + leaked, cfg)


def test_does_not_modify_input():
    df = _df([(0, 0, 10.0)])
    make_labels(df, THR)
    assert list(df.columns) == ["Cancelled", "Diverted", "ArrDelay"]
