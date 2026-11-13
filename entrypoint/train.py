"""Huấn luyện mô hình theo config (mặc định LightGBM, bộ đặc trưng schedule_v0) và đánh giá trên val.

Chạy từ thư mục gốc:  python -m entrypoint.train [--overwrite]
Test KHÔNG được dùng ở bước này.
"""
import argparse
import sys

from src.pipelines import training_pipeline
from src.pipelines.utils import get_path, load_config


def _section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Cho phép ghi đè thư mục run đã có")
    args = parser.parse_args(argv)

    cfg = load_config()
    mcfg = cfg["model"]
    out_dir = get_path(mcfg["runs_dir"]) / mcfg["run_name"]
    if (out_dir / "metrics.json").exists() and not args.overwrite:
        sys.exit(f"{out_dir} đã có kết quả. Đổi model.run_name trong config hoặc dùng --overwrite.")

    _section(f"Run {mcfg['run_name']} | model={mcfg['name']} | features={mcfg['feature_set']}")
    res = training_pipeline.run(cfg, out_dir=out_dir)

    ev = mcfg["eval_split"]
    m = res["metrics_" + ev]
    _section(f"Metric trên {ev} (best_iteration={res['best_iteration']})")
    print(f"n={m['n']:,} | tỉ lệ y=1 thực tế={m['pos_rate']:.4%} | xác suất dự báo TB={m['mean_pred']:.4%}")
    print(f"ROC-AUC={m['roc_auc']:.4f} | AP={m['ap']:.4f} | Brier={m['brier']:.5f}")
    b = m["baseline_const"]
    print(f"Mốc hằng số (p={b['pred']:.4f}, tỉ lệ y=1 của train): AP={b['ap']:.4f} | Brier={b['brier']:.5f}")
    print(f"Top {m['alert_budget']:.0%} ({m['n_alert']:,} chuyến, ngưỡng điểm {m['alert_threshold']:.4f}): "
          f"Precision={m['precision_at_budget']:.4f} | Recall={m['recall_at_budget']:.4f}")
    for c, v in m["components"].items():
        print(f"  {c:<13} n={v['n']:>7,} | recall top {m['alert_budget']:.0%}={v['recall_at_budget']:.4f} "
              f"| xác suất dự báo TB={v['mean_pred']:.4f}")

    _section("Feature importance (gain)")
    for r in res["feature_importance"]:
        print(f"  {r['feature']:<18} gain share={r['gain_share']:.3f} | splits={r['split']}")

    _section("Tổng kết")
    t = res["timing_seconds"]
    print(f"Thời gian: nạp {t['load']}s, huấn luyện {t['train']}s, tổng {t['total']}s | RAM đỉnh: {res['peak_rss_mb']:.0f} MB")
    print(f"Đã lưu: {out_dir} (model.txt, metrics.json, calibration.csv/png, feature_importance.csv)")
    print(f"Dự đoán {ev}: {res['outputs']['predictions']}")


if __name__ == "__main__":
    main()
