"""Pipeline huấn luyện: nạp train/val từ flights.parquet -> LightGBM -> đánh giá trên val.

Quy tắc:
- Chỉ nạp các split cần dùng (mặc định train + val); test KHÔNG được đọc ở đây.
- Chỉ dùng dòng có nhãn xác định. Không class weight, không resample, không tuning.
- Val dùng cho early stopping và báo cáo metric (vì vậy metric val hơi lạc quan).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.pipelines import feature_eng_pipeline as fe
from src.pipelines.utils import file_md5, get_path, machine_info, peak_rss_mb, rss_mb


# --------------------------------------------------------------------------- #
# Nạp dữ liệu
# --------------------------------------------------------------------------- #
def load_flights(cfg: dict, splits: list[str], columns: list[str], path: Path | None = None) -> pd.DataFrame:
    """Đọc flights.parquet, chỉ các cột và split cần dùng (lọc ngay khi đọc)."""
    path = Path(path) if path is not None else get_path(cfg["data"]["flights_file"])
    split_col = cfg["split"]["col"]
    cols = list(dict.fromkeys(columns + [split_col]))
    df = pd.read_parquet(path, columns=cols, filters=[(split_col, "in", list(splits))])
    got = set(df[split_col].astype(str).unique())
    assert got <= set(splits), f"Nạp nhầm split ngoài {splits}: {sorted(got - set(splits))}"
    return df


def needed_columns(cfg: dict, feature_set: str) -> list[str]:
    """Cột đặc trưng + nhãn + thành phần + khóa chuyến bay (khóa chỉ để lưu dự đoán)."""
    cols, _ = fe.feature_columns(cfg, feature_set)
    keys = list(cfg["data"]["dup_key"])
    return list(dict.fromkeys(cols + [cfg["target"], "label_status"] + fe.COMPONENT_COLS + keys))


# --------------------------------------------------------------------------- #
# Mô hình
# --------------------------------------------------------------------------- #
def build_model(cfg: dict):
    """Chọn hàm huấn luyện theo cfg['model']['name']."""
    name = cfg["model"]["name"]
    if name == "lightgbm":
        return train_lightgbm
    if name == "random_forest":
        # Thành viên làm Random Forest (Kaggle) bổ sung ở đây, dùng cùng load_flights / select_split / evaluate.
        raise NotImplementedError("random_forest: chưa cài đặt trong repo này")
    raise ValueError(f"Model không hỗ trợ: {name}")


def train_lightgbm(X_tr, y_tr, X_val, y_val, cfg: dict, categorical: list[str]):
    """Huấn luyện LightGBM với early stopping trên val. Trả về (booster, lịch sử metric)."""
    import lightgbm as lgb

    mcfg = cfg["model"]
    params = dict(mcfg["params"])
    params["metric"] = list(mcfg["eval_metrics"])
    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=categorical, free_raw_data=True)
    dval = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical, reference=dtrain, free_raw_data=True)
    history: dict = {}
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=mcfg["num_boost_round"],
        valid_sets=[dval],
        valid_names=[mcfg["eval_split"]],
        callbacks=[
            lgb.early_stopping(mcfg["early_stopping_rounds"], first_metric_only=True, verbose=True),
            lgb.log_evaluation(mcfg.get("log_every", 100)),
            lgb.record_evaluation(history),
        ],
    )
    return booster, history


# --------------------------------------------------------------------------- #
# Đánh giá
# --------------------------------------------------------------------------- #
def top_k_mask(scores: np.ndarray, budget: float) -> np.ndarray:
    """Đánh dấu đúng ceil(budget * n) dòng có điểm cao nhất (hòa điểm: thứ tự ổn định)."""
    n = len(scores)
    k = int(np.ceil(budget * n))
    mask = np.zeros(n, dtype=bool)
    if k > 0:
        idx = np.argsort(-scores, kind="stable")[:k]
        mask[idx] = True
    return mask


def evaluate(y, p, components: pd.DataFrame | None, budget: float, bins: int, base_rate: float | None = None):
    """Metric trên một tập: ROC-AUC, AP, Brier, top-k, calibration.

    base_rate: dự báo hằng số dùng làm mốc (thường là tỉ lệ y=1 của train).
    Trả về (dict metric, bảng calibration).
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    alert = top_k_mask(p, budget)
    tp = int((alert & (y == 1)).sum())
    out = {
        "n": int(len(y)),
        "pos_rate": float(y.mean()),
        "mean_pred": float(p.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "alert_budget": budget,
        "n_alert": int(alert.sum()),
        "alert_threshold": float(p[alert].min()) if alert.any() else float("nan"),
        "precision_at_budget": tp / int(alert.sum()) if alert.any() else float("nan"),
        "recall_at_budget": tp / int((y == 1).sum()) if (y == 1).any() else float("nan"),
    }
    if base_rate is not None:
        out["baseline_const"] = {
            "pred": float(base_rate),
            "brier": float(brier_score_loss(y, np.full(len(y), base_rate))),
            "ap": float(y.mean()),  # AP của dự báo hằng số = tỉ lệ dương
        }
    if components is not None:
        comp = {}
        for c in components.columns:
            m = components[c].to_numpy(dtype=bool)
            comp[c] = {
                "n": int(m.sum()),
                "rate": float(m.mean()),
                "recall_at_budget": float((alert & m).sum() / m.sum()) if m.any() else float("nan"),
                "mean_pred": float(p[m].mean()) if m.any() else float("nan"),
            }
        out["components"] = comp

    frac_pos, mean_pred = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    calib = pd.DataFrame({"mean_pred": mean_pred, "frac_pos": frac_pos})
    return out, calib


def save_calibration_plot(calib: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    lim = float(max(calib["mean_pred"].max(), calib["frac_pos"].max())) * 1.05
    ax.plot([0, lim], [0, lim], linestyle="--", color="grey", linewidth=1, label="hoàn hảo")
    ax.plot(calib["mean_pred"], calib["frac_pos"], marker="o", label="mô hình")
    ax.set_xlabel("Xác suất dự báo trung bình (bin)")
    ax.set_ylabel("Tỉ lệ y=1 thực tế")
    ax.set_title(title)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Chạy toàn bộ
# --------------------------------------------------------------------------- #
def run(cfg: dict, flights_path: Path | None = None, out_dir: Path | None = None,
        pred_dir: Path | None = None, verbose: bool = True) -> dict:
    """Nạp train/val -> huấn luyện -> đánh giá val -> lưu model, metric, dự đoán val."""
    t_start = time.perf_counter()
    mcfg, ecfg = cfg["model"], cfg["evaluation"]
    feature_set = mcfg["feature_set"]
    tr_name, ev_name = mcfg["train_split"], mcfg["eval_split"]
    assert "test" not in (tr_name, ev_name), "Không dùng test ở bước huấn luyện"
    log = print if verbose else (lambda *a, **k: None)

    flights_path = Path(flights_path) if flights_path is not None else get_path(cfg["data"]["flights_file"])
    out_dir = Path(out_dir) if out_dir is not None else get_path(mcfg["runs_dir"]) / mcfg["run_name"]
    pred_dir = Path(pred_dir) if pred_dir is not None else get_path(mcfg["predictions_dir"])

    # ---- Nạp ----
    t0 = time.perf_counter()
    df = load_flights(cfg, [tr_name, ev_name], needed_columns(cfg, feature_set), flights_path)
    cols, categorical = fe.feature_columns(cfg, feature_set)
    X_tr, y_tr, c_tr = fe.select_split(df, cfg, tr_name, feature_set)
    X_val, y_val, c_val = fe.select_split(df, cfg, ev_name, feature_set)
    keys_val = df.loc[X_val.index, list(cfg["data"]["dup_key"])].reset_index(drop=True)
    del df
    load_s = time.perf_counter() - t0
    data_info = {
        "flights_file": str(flights_path.name),
        "flights_md5": file_md5(flights_path),
        "features": cols,
        "categorical": categorical,
        tr_name: {"rows": int(len(y_tr)), "y1_rate": float(y_tr.mean())},
        ev_name: {"rows": int(len(y_val)), "y1_rate": float(y_val.mean())},
    }
    log(f"Nạp xong {load_s:.1f}s | {tr_name}: X={X_tr.shape}, y=1 {y_tr.mean():.4%} | "
        f"{ev_name}: X={X_val.shape}, y=1 {y_val.mean():.4%} | RSS={rss_mb():.0f} MB, đỉnh={peak_rss_mb():.0f} MB")

    # ---- Huấn luyện ----
    train_fn = build_model(cfg)
    t0 = time.perf_counter()
    booster, history = train_fn(X_tr, y_tr, X_val, y_val, cfg, categorical)
    train_s = time.perf_counter() - t0
    log(f"Huấn luyện xong {train_s:.1f}s | best_iteration={booster.best_iteration} | "
        f"RSS={rss_mb():.0f} MB, đỉnh={peak_rss_mb():.0f} MB")

    # ---- Đánh giá trên val ----
    p_val = booster.predict(X_val, num_iteration=booster.best_iteration)
    metrics, calib = evaluate(y_val, p_val, c_val, ecfg["alert_budget"], ecfg["calibration_bins"],
                              base_rate=float(y_tr.mean()))

    importance = pd.DataFrame({
        "feature": booster.feature_name(),
        "gain": booster.feature_importance("gain"),
        "split": booster.feature_importance("split"),
    }).sort_values("gain", ascending=False)
    importance["gain_share"] = importance["gain"] / importance["gain"].sum()

    # ---- Lưu ----
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    # Không dùng booster.save_model: hàm C++ của LightGBM không mở được đường dẫn có dấu (vd "Học tập")
    # trên Windows. Ghi chuỗi model bằng Python; đọc lại bằng lgb.Booster(model_str=...).
    (out_dir / "model.txt").write_text(
        booster.model_to_string(num_iteration=booster.best_iteration), encoding="utf-8")
    calib.to_csv(out_dir / "calibration.csv", index=False)
    save_calibration_plot(calib, out_dir / "calibration.png", f"Calibration ({ev_name}) - {mcfg['run_name']}")
    importance.to_csv(out_dir / "feature_importance.csv", index=False)
    pred_path = pred_dir / f"{ev_name}_{mcfg['run_name']}.parquet"
    # Khóa chuyến bay (dup_key, không trùng) để ghép lại với flights.parquet khi cần
    preds = pd.concat([
        keys_val,
        pd.DataFrame({"y": y_val.to_numpy(), "score": p_val.astype("float32")}),
        c_val.reset_index(drop=True),
    ], axis=1)
    preds.to_parquet(pred_path, index=False)

    total_s = time.perf_counter() - t_start
    result = {
        "run_name": mcfg["run_name"],
        "model": mcfg["name"],
        "feature_set": feature_set,
        "params": mcfg["params"],
        "num_boost_round": mcfg["num_boost_round"],
        "early_stopping_rounds": mcfg["early_stopping_rounds"],
        "best_iteration": int(booster.best_iteration),
        "best_score_val": {k: float(v) for k, v in booster.best_score.get(ev_name, {}).items()},
        "data": data_info,
        "metrics_" + ev_name: metrics,
        "feature_importance": importance.to_dict(orient="records"),
        "timing_seconds": {"load": round(load_s, 1), "train": round(train_s, 1), "total": round(total_s, 1)},
        "peak_rss_mb": round(peak_rss_mb(), 0),
        "machine": machine_info(),
        "outputs": {"model": str(out_dir / "model.txt"), "predictions": str(pred_path)},
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    return result
