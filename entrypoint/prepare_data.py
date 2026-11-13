"""Bước chung: nạp BTS -> nhãn -> chia tập -> kiểm tra -> lưu parquet.

Chạy từ thư mục gốc:  python -m entrypoint.prepare_data [--overwrite]
"""
import argparse
import json
import sys
import time

import pandas as pd
import pyarrow.parquet as pq

from src.pipelines import data_pipeline as dp
from src.pipelines.utils import file_md5, get_path, load_config, machine_info, peak_rss_mb, ram_status, rss_mb


def _section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Cho phép ghi đè flights.parquet đã có")
    args = parser.parse_args(argv)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    t_start = time.perf_counter()
    cfg = load_config()
    out_path = get_path(cfg["data"]["flights_file"])
    report_path = get_path(cfg["data"]["report_file"])
    date_col = cfg["data"]["date_col"]
    split_cfg = cfg["split"]

    if out_path.exists() and not args.overwrite:
        sys.exit(f"{out_path} đã tồn tại. Bản chia chỉ tạo một lần; dùng --overwrite nếu thật sự muốn tạo lại.")

    _section("Cấu hình máy")
    machine = machine_info()
    for k, v in machine.items():
        print(f"{k}: {v}")

    # ---- Nạp ----
    _section("Nạp từng tháng")
    files = dp.list_raw_files(cfg)
    print(f"Đủ {len(files)} tháng: {[f'{y}-{m:02d}' for y, m in files]}")
    df, month_report = dp.load_all_months(cfg)
    print(f"\nSau khi nối: shape={df.shape}, bộ nhớ DataFrame={df.memory_usage(deep=True).sum() / 2**20:.0f} MB, "
          f"RSS={rss_mb():.0f} MB, đỉnh={peak_rss_mb():.0f} MB")
    print(df.dtypes.to_string())

    _section("Thống kê CRSDepTime/CRSArrTime theo tháng")
    print(month_report[["month", "rows", "CRSDepTime_2400", "CRSArrTime_2400",
                        "CRSDepTime_invalid", "CRSArrTime_invalid"]].to_string(index=False))

    _section("Trùng lặp")
    print(month_report[["month", "dup_full_rows", "dup_key_rows"]].to_string(index=False))
    n_dup_full = int(month_report["dup_full_rows"].sum())
    n_dup_key = int(month_report["dup_key_rows"].sum())
    print(f"Tổng trùng toàn phần: {n_dup_full}; trùng theo khóa {cfg['data']['dup_key']}: {n_dup_key}")
    if n_dup_full > 0:
        sys.exit("[DỪNG] Có dòng trùng toàn phần. Không tự xóa; cần hỏi lại cách xử lý trước khi lưu.")

    # ---- Nhãn ----
    _section("Tạo nhãn")
    thr = cfg["label"]["delay_threshold_min"]
    df = dp.make_labels(df, thr, copy=False)
    print(df["label_status"].value_counts().reindex(dp.LABEL_STATUSES).to_string())
    assert df["y"].dtype == "Int8"
    assert df["y"].isna().sum() == (df["label_status"] == "undetermined").sum()

    _section("ArrDelay thiếu so với hủy + chuyển hướng (theo tháng)")
    miss = dp.arrdelay_missing_check(df, date_col)
    print(miss.to_string())
    mismatch = miss[miss["diff"] != 0]
    if len(mismatch):
        print(f"[CẢNH BÁO] Lệch ở {len(mismatch)} tháng: {list(mismatch.index)}")
    else:
        print("OK: mọi tháng ArrDelay thiếu == hủy + chuyển hướng.")

    # ---- Chia tập ----
    _section("Chia tập theo thời gian")
    df = dp.assign_split(df, split_cfg, date_col, copy=False)
    seg_df = dp.validate_split(df, split_cfg, date_col)
    print(seg_df.to_string(index=False))
    print("OK: các phần không giao nhau, thứ tự thời gian đúng, mọi dòng có split.")
    summary = dp.split_summary(df, split_cfg, date_col)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4%}"))

    # ---- Lưu ----
    _section("Lưu parquet")
    dp.check_no_forbidden(list(df.columns), cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    df.to_parquet(out_path, engine="pyarrow", index=False)
    n_saved = pq.ParquetFile(out_path).metadata.num_rows
    assert n_saved == len(df), "Số dòng trong parquet không khớp"
    md5 = file_md5(out_path)
    cfg_md5 = file_md5(get_path("config/config.yaml"))
    print(f"Đã lưu {out_path} ({out_path.stat().st_size / 2**20:.1f} MB, {n_saved:,} dòng, "
          f"{time.perf_counter() - t0:.1f}s)")
    print(f"md5 flights.parquet: {md5}")
    print(f"md5 config.yaml:     {cfg_md5}")

    total_s = time.perf_counter() - t_start
    peak = peak_rss_mb()
    report = {
        "machine": machine,
        "ram_after": ram_status(),
        "total_seconds": round(total_s, 1),
        "peak_rss_mb": round(peak, 0),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "flights_parquet_md5": md5,
        "config_md5": cfg_md5,
        "dup_full_rows": n_dup_full,
        "dup_key_rows": n_dup_key,
        "months": month_report.to_dict(orient="records"),
        "arrdelay_missing_check": miss.reset_index().to_dict(orient="records"),
        "split_segments": json.loads(seg_df.to_json(orient="records", date_format="iso")),
        "split_summary": json.loads(summary.to_json(orient="records", date_format="iso")),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"Đã ghi báo cáo {report_path}")

    _section("Tổng kết")
    print(f"Thời gian: {total_s:.1f}s | RAM đỉnh tiến trình: {peak:.0f} MB")


if __name__ == "__main__":
    main()
