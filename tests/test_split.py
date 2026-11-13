"""Test chia tập theo thời gian (mốc đọc từ config/config.yaml)."""
import copy

import pandas as pd
import pytest

from src.pipelines.data_pipeline import assign_split, split_names, validate_split, validate_split_config
from src.pipelines.utils import load_config

SPLIT_CFG = load_config()["split"]


def _df(dates):
    return pd.DataFrame({"FlightDate": pd.to_datetime(dates)})


@pytest.mark.parametrize(
    "date, expected",
    [
        ("2024-12-01", "warmup"),
        ("2024-12-31", "warmup"),
        ("2025-01-01", "train"),
        ("2025-08-24", "train"),
        ("2025-08-25", "gap"),
        ("2025-08-31", "gap"),
        ("2025-09-01", "val"),
        ("2025-10-24", "val"),
        ("2025-10-25", "gap"),
        ("2025-10-31", "gap"),
        ("2025-11-01", "test"),
        ("2025-12-31", "test"),
    ],
)
def test_boundaries(date, expected):
    out = assign_split(_df([date]), SPLIT_CFG)
    assert out[SPLIT_CFG["col"]].iloc[0] == expected


@pytest.mark.parametrize("date", ["2024-11-30", "2026-01-01"])
def test_date_outside_ranges_raises(date):
    with pytest.raises(ValueError):
        assign_split(_df([date]), SPLIT_CFG)


def test_split_names_order():
    assert split_names(SPLIT_CFG) == ["warmup", "train", "gap", "val", "test"]


def test_full_period_disjoint_and_ordered():
    dates = pd.date_range("2024-12-01", "2025-12-31", freq="D")
    out = assign_split(_df(dates), SPLIT_CFG)
    validate_split(out, SPLIT_CFG)  # assert bên trong
    col = SPLIT_CFG["col"]
    # mỗi ngày đúng một split
    assert out[col].notna().all()
    by = out.groupby(col, observed=True)["FlightDate"].agg(["min", "max", "count"])
    assert by.loc["train", "max"] < by.loc["val", "min"] < by.loc["test", "min"]
    assert by.loc["warmup", "max"] < by.loc["train", "min"]
    assert by.loc["val", "max"] < by.loc["test", "min"]
    assert by.loc["gap", "count"] == 14  # hai khoảng đệm 7 ngày
    sets = {s: set(out.loc[out[col] == s, "FlightDate"]) for s in split_names(SPLIT_CFG)}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (sets[a] & sets[b])


def test_overlapping_config_rejected():
    bad = copy.deepcopy(SPLIT_CFG)
    bad["ranges"][1]["end"] = "2025-08-26"  # train chồng lên gap
    with pytest.raises(AssertionError):
        validate_split_config(bad)


def test_unordered_config_rejected():
    bad = copy.deepcopy(SPLIT_CFG)
    bad["ranges"][3], bad["ranges"][5] = bad["ranges"][5], bad["ranges"][3]  # đảo val và test
    with pytest.raises(AssertionError):
        validate_split_config(bad)


def test_validate_detects_wrong_label():
    out = assign_split(_df(["2025-03-01", "2025-11-15"]), SPLIT_CFG)
    out[SPLIT_CFG["col"]] = pd.Categorical(["test", "train"], categories=split_names(SPLIT_CFG))
    with pytest.raises(AssertionError):
        validate_split(out, SPLIT_CFG)
