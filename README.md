# Airline – CS114 ML Project

Dự báo rủi ro gián đoạn chuyến bay (hủy, chuyển hướng, hoặc đến trễ ≥ 60 phút) tại T-24h, dữ liệu BTS 12/2024–12/2025. Hai mô hình: LightGBM và Random Forest. Mô tả bài toán: [docs/flight_disruption.md](docs/flight_disruption.md).

## Cấu trúc
```
CLAUDE.md                     # hướng dẫn làm việc cho Claude Code
README.md
requirements.txt              # phiên bản thư viện (phải giống nhau giữa local và Kaggle)
config/config.yaml            # đường dẫn, usecols, dtype, ngưỡng 60 phút, mốc chia tập, seed
docs/
  flight_disruption.md        # mô tả bài toán của nhóm
  README.md                   # hướng dẫn dùng dữ liệu chung cho thành viên kia
data/                         # không commit (dữ liệu gửi riêng bằng zip)
  01-raw/                     # 13 file zip BTS theo tháng (không sửa)
  02-preprocessed/            # flights.parquet (nhãn + split), prepare_data_report.json
  03-features/                # đặc trưng (giai đoạn sau)
  04-predictions/             # kết quả dự đoán (giai đoạn sau)
entrypoint/
  prepare_data.py             # bước chung: nạp -> nhãn -> chia tập -> kiểm tra -> lưu parquet
  train.py                    # huấn luyện LightGBM, đánh giá trên val
  inference.py                # dự đoán (giai đoạn sau, đang là template)
models/                       # model đã train (giai đoạn sau)
notebooks/
  EDA.ipynb                   # EDA bước chung
  Baseline.ipynb              # giai đoạn sau
  figures/                    # hình EDA (png)
src/pipelines/
  utils.py                    # load config, đường dẫn, md5, đo RAM, cấu hình máy
  data_pipeline.py            # nạp từng tháng, tạo nhãn, gán split, kiểm tra
  feature_eng_pipeline.py     # chọn đặc trưng (schedule_v0), tách split
  training_pipeline.py        # LightGBM: nạp train/val, huấn luyện, đánh giá
  inference_pipeline.py       # giai đoạn sau (template)
tests/
  test_labels.py              # quy tắc nhãn 3 trường hợp, cột cấm
  test_split.py               # chia tập theo thời gian, ranh giới ngày
  test_training.py            # chống rò rỉ, không đọc test, metric, chạy end-to-end
```

## Cài đặt
```bash
pip install -r requirements.txt
```

## Chạy (từ thư mục gốc project)
1. Đặt 13 file zip BTS (`..._2024_12.zip` đến `..._2025_12.zip`) vào `data/01-raw/`.
2. Bước chung: `python -m entrypoint.prepare_data` (thêm `--overwrite` nếu thật sự muốn tạo lại `flights.parquet`). Lệnh in số dòng, tỉ lệ y=1 từng split, md5 và ghi `data/02-preprocessed/prepare_data_report.json`.
3. Test: `python -m pytest tests`
4. EDA: `jupyter nbconvert --to notebook --execute --inplace notebooks/EDA.ipynb` (hình lưu ở `notebooks/figures/`).
5. Huấn luyện LightGBM: `python -m entrypoint.train` (thêm `--overwrite` để chạy lại cùng `model.run_name`). Đọc mục `features`, `model`, `evaluation` trong config; chỉ nạp train + val, early stopping và báo metric trên val (test không dùng). Kết quả lưu ở `models/<run_name>/` (model.txt, metrics.json, calibration, feature importance) và dự đoán val ở `data/04-predictions/`.
6. Dự đoán (`python -m entrypoint.inference`): giai đoạn sau, đang là template.

## Dữ liệu dùng chung
`data/02-preprocessed/flights.parquet` + `prepare_data_report.json` không nằm trong repo, được gửi riêng bằng file zip (giải nén ở thư mục gốc repo; bước 1–2 chỉ cần khi muốn tự tạo lại). Đối chiếu bằng số dòng, tỉ lệ y=1 từng split và md5 (xem [docs/README.md](docs/README.md), mục 3; md5 parquet chỉ trùng khi cùng phiên bản pyarrow).
