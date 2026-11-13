"""Test giai đoạn huấn luyện: chọn đặc trưng, tách split, metric, và không đọc test."""
import copy

import numpy as np
import pandas as pd
import pytest

from src.pipelines import feature_eng_pipeline as fe
from src.pipelines import training_pipeline as tp
from src.pipelines.utils import load_config

CFG = load_config()


# --------------------------------------------------------------------------- #
# Đặc trưng
# --------------------------------------------------------------------------- #
def test_feature_columns_no_leak():
    cols, cat = fe.feature_columns(CFG, CFG["model"]["feature_set"])
    banned = (set(CFG["forbidden_cols"]) | set(CFG["label_only_cols"])
              | set(CFG["features"]["never_features"]))
    assert not set(cols) & banned
    assert "Month" not in cols and "Flight_Number_Reporting_Airline" not in cols
    assert set(cat) <= set(cols)


@pytest.mark.parametrize("bad", ["ArrDelay", "DepDelay", "y", "Month", "DivAirportLandings", "split"])
def test_feature_columns_rejects_banned(bad):
    cfg = copy.deepcopy(CFG)
    cfg["features"]["tmp"] = {"categorical": [], "numeric": ["Distance", bad]}
    with pytest.raises(AssertionError):
        fe.feature_columns(cfg, "tmp")


# --------------------------------------------------------------------------- #
# Dữ liệu giả có cấu trúc giống flights.parquet
# --------------------------------------------------------------------------- #
def _fake_flights(n_per_split: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = {"train": "2025-03-10", "gap": "2025-08-27", "val": "2025-09-15", "test": "2025-11-20"}
    parts = []
    for split, d in dates.items():
        n = n_per_split
        dep = rng.integers(500, 2300, n).astype("int16")
        p = np.where(dep > 1500, 0.35, 0.05)
        y = (rng.random(n) < p).astype(int)
        part = pd.DataFrame({
            "FlightDate": pd.Timestamp(d),
            "Reporting_Airline": rng.choice(["AA", "DL", "WN"], n),
            "Flight_Number_Reporting_Airline": np.arange(n, dtype="int16"),
            "Origin": rng.choice(["ATL", "ORD", "DEN"], n),
            "Dest": rng.choice(["LAX", "SEA"], n),
            "CRSDepTime": dep,
            "CRSArrTime": ((dep + 200) % 2400).astype("int16"),
            "CRSElapsedTime": rng.uniform(60, 300, n).astype("float32"),
            "Distance": rng.uniform(200, 2500, n).astype("float32"),
            "DayOfWeek": rng.integers(1, 8, n).astype("int8"),
            "Month": np.int8(pd.Timestamp(d).month),
            "is_cancelled": False,
            "is_diverted": False,
            "is_delay60": y.astype(bool),
            "label_status": np.where(y == 1, "positive", "negative"),
            "y": pd.array(y, dtype="Int8"),
            "split": split,
        })
        if split == "test":
            # Dòng test "độc": sân bay riêng, toàn nhãn 1. Nếu bị đọc thì test sẽ phát hiện.
            part["Origin"] = "POISON"
            part["y"] = pd.array(np.ones(n, dtype=int), dtype="Int8")
            part["label_status"] = "positive"
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    # 1 dòng undetermined trong val
    i = df.index[df["split"] == "val"][0]
    df.loc[i, "y"] = pd.NA
    df.loc[i, "label_status"] = "undetermined"
    df.loc[i, "is_delay60"] = False
    for c in ["Reporting_Airline", "Origin", "Dest", "label_status"]:
        df[c] = df[c].astype("category")
    df["split"] = pd.Categorical(df["split"], categories=["warmup", "train", "gap", "val", "test"])
    return df


def _small_cfg() -> dict:
    cfg = copy.deepcopy(CFG)
    cfg["model"]["num_boost_round"] = 30
    cfg["model"]["early_stopping_rounds"] = 5
    cfg["model"]["log_every"] = 0
    cfg["model"]["params"]["num_threads"] = 1
    return cfg


def test_select_split_drops_undetermined_and_other_splits():
    df = _fake_flights()
    X, y, comp = fe.select_split(df, CFG, "val", "schedule_v0")
    assert len(X) == (df["split"] == "val").sum() - 1
    assert y.dtype == "int8" and y.notna().all()
    assert list(comp.columns) == fe.COMPONENT_COLS
    cols, _ = fe.feature_columns(CFG, "schedule_v0")
    assert list(X.columns) == cols


def test_load_flights_never_reads_test(tmp_path):
    path = tmp_path / "flights.parquet"
    _fake_flights().to_parquet(path, index=False)
    df = tp.load_flights(CFG, ["train", "val"], tp.needed_columns(CFG, "schedule_v0"), path)
    assert set(df["split"].astype(str)) == {"train", "val"}
    assert "POISON" not in set(df["Origin"].astype(str))


# --------------------------------------------------------------------------- #
# Metric
# --------------------------------------------------------------------------- #
def test_top_k_and_evaluate_by_hand():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    p = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.05])
    comp = pd.DataFrame({"is_cancelled": y == 1, "is_diverted": False, "is_delay60": False})
    m, calib = tp.evaluate(y, p, comp, budget=0.2, bins=2, base_rate=0.3)
    # top 20% = 2 dòng có điểm cao nhất: 0.9 (y=1) và 0.8 (y=0)
    assert m["n_alert"] == 2
    assert m["precision_at_budget"] == pytest.approx(0.5)
    assert m["recall_at_budget"] == pytest.approx(1 / 3)
    assert m["components"]["is_cancelled"]["recall_at_budget"] == pytest.approx(1 / 3)
    assert m["baseline_const"]["ap"] == pytest.approx(0.3)
    assert 0 <= m["roc_auc"] <= 1 and len(calib) <= 2


# --------------------------------------------------------------------------- #
# Chạy toàn bộ trên dữ liệu giả
# --------------------------------------------------------------------------- #
def test_run_end_to_end_uses_only_train_and_val(tmp_path):
    df = _fake_flights()
    path = tmp_path / "flights.parquet"
    df.to_parquet(path, index=False)
    res = tp.run(_small_cfg(), flights_path=path, out_dir=tmp_path / "run",
                 pred_dir=tmp_path / "pred", verbose=False)

    assert res["data"]["train"]["rows"] == (df["split"] == "train").sum()
    assert res["data"]["val"]["rows"] == (df["split"] == "val").sum() - 1
    m = res["metrics_val"]
    assert m["roc_auc"] > 0.6  # có tín hiệu thật theo giờ đi
    assert res["best_iteration"] >= 1
    for f in ["model.txt", "metrics.json", "calibration.csv", "calibration.png", "feature_importance.csv"]:
        assert (tmp_path / "run" / f).exists()

    # Model đọc lại được và cho cùng dự đoán
    import lightgbm as lgb
    booster = lgb.Booster(model_str=(tmp_path / "run" / "model.txt").read_text(encoding="utf-8"))
    assert booster.num_trees() == res["best_iteration"]

    preds = pd.read_parquet(tmp_path / "pred" / f"val_{CFG['model']['run_name']}.parquet")
    assert len(preds) == res["data"]["val"]["rows"]
    assert (preds["FlightDate"] == pd.Timestamp("2025-09-15")).all()
    assert preds["score"].between(0, 1).all()
