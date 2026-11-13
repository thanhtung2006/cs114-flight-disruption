# Hướng dẫn dùng dữ liệu chung (cho thành viên làm Random Forest trên Kaggle)

File này mô tả dữ liệu đã chuẩn bị ở **bước chung** (nạp dữ liệu, tạo nhãn, chia tập theo thời gian, EDA) để hai mô hình LightGBM và Random Forest dùng **cùng tập mẫu, cùng nhãn, cùng cách chia**. Mô tả bài toán đầy đủ: [flight_disruption.md](flight_disruption.md).

> **Trạng thái: v1, bản nền.** Mốc chia tập vẫn là **đề xuất**, phạm vi sân bay **chưa chốt**, **chưa có đặc trưng** (lịch sử 7/30 ngày, T-24h, múi giờ). Nếu mốc chia thay đổi, file parquet và md5 sẽ đổi, khi đó sẽ có version mới.

## 1. Repo có gì, dữ liệu lấy ở đâu

Repo chỉ chứa **code và tài liệu**. Thư mục `data/` không được commit (chỉ có `.gitkeep`). Dữ liệu đã tạo sẵn được **gửi riêng** bằng file `cs114_data_v1.zip`:

| Trong zip | Nội dung |
|---|---|
| `data/02-preprocessed/flights.parquet` | 7.592.200 dòng, 21 cột, 69 MB |
| `data/02-preprocessed/prepare_data_report.json` | số dòng, tỉ lệ y=1, md5, cấu hình máy lúc tạo |
| `MD5SUMS.txt` | md5 của 2 file trên |

**Cách dùng:** giải nén zip **ở thư mục gốc repo** (nơi có `config/`, `src/`), 2 file sẽ vào đúng `data/02-preprocessed/`. Sau đó kiểm tra md5 và số dòng như mục 3.

**Dùng trên Kaggle:** upload `flights.parquet` (và `config.yaml` nếu cần) lên một Kaggle Dataset private của bạn, rồi đọc từ `/kaggle/input/<tên-dataset>/flights.parquet`.

**Tự tạo lại** (không bắt buộc, để kiểm tra quy trình): xem các bước bên dưới. Nếu đã giải nén zip thì `prepare_data` sẽ từ chối ghi đè parquet, phải thêm `--overwrite`.

| File trong repo | Nội dung |
|---|---|
| [config/config.yaml](../config/config.yaml) | đường dẫn, cột nạp, mốc chia, ngưỡng 60 phút, danh sách cột cấm, seed 42 |
| [requirements.txt](../requirements.txt) | phiên bản thư viện, cần cài **giống nhau** ở hai bên |
| [src/pipelines/data_pipeline.py](../src/pipelines/data_pipeline.py) | nạp từng tháng, tạo nhãn, gán split, kiểm tra |
| [entrypoint/prepare_data.py](../entrypoint/prepare_data.py) | chạy toàn bộ bước chung, ghi parquet + báo cáo |
| [tests/](../tests/) | test nhãn và chia tập (`python -m pytest tests`) |
| [notebooks/EDA.ipynb](../notebooks/EDA.ipynb), [notebooks/figures/](../notebooks/figures/) | EDA đã chạy sẵn (xem được output, không cần chạy lại) |

**Tự tạo dữ liệu từ file BTS gốc:**

1. Tải 13 file prezipped *Reporting Carrier On-Time Performance (1987-present)* từ [BTS TranStats](https://transtats.bts.gov/ONTIME/), tháng 12/2024 đến 12/2025. Tên file phải có dạng `On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2025_1.zip` (năm_tháng ở cuối, không cần giải nén).
2. Đặt cả 13 file vào `data/01-raw/`.
3. Từ thư mục gốc repo: `python -m entrypoint.prepare_data`. Lần chạy local mất 64,5 giây, RAM đỉnh khoảng 1,4 GB.
4. Kết quả: `data/02-preprocessed/flights.parquet` (7.592.200 dòng, 21 cột, khoảng 69 MB) và `data/02-preprocessed/prepare_data_report.json`.

**Chạy trên Kaggle:** upload 13 file zip thành một input riêng, clone repo vào `/kaggle/working`, rồi sửa `data.raw_dir` trong `config.yaml` thành đường dẫn tuyệt đối của input (vd `/kaggle/input/<tên-input>`). Các đường dẫn tuyệt đối được dùng nguyên; file đầu ra vẫn ghi vào `data/02-preprocessed/` trong repo. Lưu ý: sửa config thì md5 của `config.yaml` sẽ khác, chỉ cần các mục còn lại không đổi.

## 2. Môi trường

Phiên bản local (Python 3.13.5, Windows 11): `pandas==2.3.2`, `numpy==2.4.4`, `pyarrow==23.0.1`, `scikit-learn==1.7.2`, `lightgbm==4.6.0`.

- Trên Kaggle cần cài cùng phiên bản `pandas`, `scikit-learn`, `lightgbm` (numpy 2.4 yêu cầu Python ≥ 3.11).
- Thời gian và RAM đo trên Kaggle **không so trực tiếp** với local; luôn ghi kèm cấu hình máy.

## 3. Kiểm tra sau khi tạo dữ liệu

`prepare_data` tự in số dòng, tỉ lệ y=1 từng split và md5. Đối chiếu với bản gốc (tạo local):

- md5 `flights.parquet`: `9b1503c67beb97fc4f3a6e309d6de3c5`
- md5 `prepare_data_report.json` (bản trong zip): `192993f09413f5150772bc669cb9b615`
- md5 `config.yaml` lúc tạo parquet: `8810cbd9c87a397f417a7556bca6c7a8`. Sau đó config được thêm các mục `features`, `model`, `evaluation` (giai đoạn huấn luyện, phần `model` là của LightGBM) nên md5 config hiện tại khác. Các mục dùng để tạo parquet (`data`, `label`, `split`, `forbidden_cols`) **không đổi**, vì vậy md5 của parquet vẫn như trên.
- Số dòng và tỉ lệ từng phần: bảng ở mục 6.

Hoặc kiểm tra lại bằng tay:

```python
import pandas as pd

df = pd.read_parquet("data/02-preprocessed/flights.parquet")
print(df.shape)  # (7592200, 21)
det = df[df["label_status"] != "undetermined"]
print(det.groupby("split", observed=True)["y"].agg(["size", "mean"]))
```

md5 của parquet chỉ trùng khi cùng phiên bản pandas/pyarrow. Nếu md5 khác nhưng số dòng và tỉ lệ từng phần khớp mục 6 thì dữ liệu vẫn giống nhau; nếu số liệu khác thì báo lại để kiểm tra.

## 4. Các cột

Mỗi dòng là **một chuyến bay**. Dữ liệu gốc BTS chỉ được lọc cột và ép kiểu; **chưa điền thiếu, chưa mã hóa, chưa tạo đặc trưng**.

**Lịch bay (12 cột, được dùng làm đầu vào)**

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `FlightDate` | datetime | ngày bay (giờ địa phương) |
| `Reporting_Airline` | category | mã hãng (14 hãng) |
| `Flight_Number_Reporting_Airline` | int16 | số hiệu chuyến |
| `Origin`, `Dest` | category | sân bay đi, sân bay đến |
| `CRSDepTime`, `CRSArrTime` | int16 | giờ đi/đến dự kiến dạng `hhmm` (vd `815` = 08:15), giờ địa phương |
| `CRSElapsedTime` | float32 | thời gian bay dự kiến (phút) |
| `Distance` | float32 | khoảng cách (dặm) |
| `DayOfWeek` | int8 | 1 = thứ Hai ... 7 = Chủ nhật |
| `Month` | int8 | tháng |

**Không dùng làm đầu vào**

| Cột | Lý do |
|---|---|
| `Tail_Number` | chưa chắc đã biết trước tại T-24h; chỉ giữ lại |
| `Cancelled`, `Diverted`, `ArrDelay` | kết quả sau khi bay, **chỉ dùng để tạo nhãn** |
| `y`, `label_status`, `is_cancelled`, `is_diverted`, `is_delay60` | nhãn và thành phần nhãn |
| `split` | cột chia tập |

**Nhãn**

| Cột | Ý nghĩa |
|---|---|
| `y` (Int8, cho phép NA) | 1 = gián đoạn, 0 = bình thường, NA = chưa xác định |
| `label_status` | `positive` / `negative` / `undetermined` |
| `is_cancelled`, `is_diverted`, `is_delay60` | thành phần của lớp dương; không giao nhau (`is_delay60` = trễ ≥ 60 **và** không hủy/chuyển hướng) |

Quy tắc nhãn (không điền `ArrDelay` thiếu bằng 0):
- `y = 1` nếu `Cancelled == 1` hoặc `Diverted == 1` hoặc `ArrDelay >= 60`.
- `y = 0` nếu không hủy, không chuyển hướng, `ArrDelay` hợp lệ và `< 60`.
- Còn lại: chưa xác định (`y` = NA). Hiện chỉ có **1 dòng**: 2025-09-06, YX 5859 JFK→MVY (thuộc val), không hủy, không chuyển hướng nhưng thiếu `ArrDelay`.

## 5. Quy tắc bắt buộc

1. **Huấn luyện:** chỉ dùng `split == "train"` và `label_status != "undetermined"`.
2. **Validation** (early stopping, chọn ngưỡng, calibration): `split == "val"`. **Test** chỉ dùng **một lần** ở cuối.
3. Nhãn của `warmup` và `gap` **không** dùng để huấn luyện hay đánh giá. Các chuyến này chỉ để tính đặc trưng lịch sử cho chuyến sau (giai đoạn sau).
4. **Không dùng nhãn của val/test** để fit bất cứ thứ gì (chọn sân bay, target encoding, ngưỡng...).
5. **Không chia lại** dữ liệu, không chia ngẫu nhiên; dùng đúng cột `split`.
6. Không xử lý mất cân bằng (không class weight, không resample). Seed `42`.
7. Không dùng các cột kết quả thực tế / nguyên nhân trễ (danh sách `forbidden_cols` trong `config.yaml`).

## 6. Số liệu từng phần

| split | FlightDate | số chuyến | y=1 | hủy | chuyển hướng | trễ ≥ 60 |
|---|---|---|---|---|---|---|
| warmup | 2024-12-01 đến 2024-12-31 | 590.581 | 8,21% | 0,70% | 0,27% | 7,24% |
| train | 2025-01-01 đến 2025-08-24 | 4.550.899 | 10,39% | 1,59% | 0,31% | 8,49% |
| gap | 2025-08-25 đến 2025-08-31 và 2025-10-25 đến 2025-10-31 | 261.603 | 8,53% | 1,00% | 0,35% | 7,18% |
| val | 2025-09-01 đến 2025-10-24 | 1.036.263 (1 dòng chưa xác định) | 6,15% | 0,44% | 0,18% | 5,53% |
| test | 2025-11-01 đến 2025-12-31 | 1.152.854 | 10,93% | 2,01% | 0,21% | 8,70% |
| **Tổng** | | **7.592.200** | **9,66%** | 1,41% | 0,27% | 7,97% |

Tỉ lệ tính trên các dòng có nhãn xác định.

## 7. Lưu ý từ EDA

- **Val lệch phân phối:** y=1 chỉ 6,15% (tháng 9–10/2025 là giai đoạn thấp nhất trong 13 tháng), trong khi train 10,39% và test 10,93%. Ảnh hưởng tới early stopping, chọn ngưỡng và calibration.
- **Test rơi vào mùa lễ, đầu đông** (11–12/2025), khác mùa với phần lớn train.
- **Dòng không sắp xếp theo ngày** (giữ thứ tự file BTS: theo tháng, rồi hãng, số hiệu). Khi tính đặc trưng lịch sử phải tự sắp xếp theo thời gian.
- **Giá trị thiếu ở cột đầu vào:** chỉ `CRSElapsedTime` thiếu 2 dòng (train). `Tail_Number` thiếu 13.122 dòng (không dùng). `ArrDelay` thiếu 127.884 dòng, bằng hủy + chuyển hướng (127.883) cộng 1 dòng chưa xác định.
- **Giá trị `2400`:** 1 dòng `CRSDepTime` (7/2025), 2 dòng `CRSArrTime` (8/2025). Không có giá trị `hhmm` sai định dạng. Chưa xử lý ở bước này.
- **Không có dòng trùng** (toàn phần và theo khóa chuyến bay), không có ngày nào thiếu dữ liệu.
- **Biến phân loại:** `Reporting_Airline`, `Origin`, `Dest` có kiểu `category` (349 sân bay trong train). LightGBM dùng trực tiếp được; Random Forest của scikit-learn không nhận `category`, cần mã hóa. Cách mã hóa nên thống nhất giữa hai người; nếu dùng target encoding thì chỉ fit trên train.
- **Không cần chuẩn hóa/scale** vì cả hai mô hình đều dựa trên cây.

## 8. Việc tiếp theo (chưa làm)

- Chốt mốc chia và phạm vi sân bay (chỉ dựa trên train).
- Bảng múi giờ sân bay, tính mốc T-24h theo UTC.
- Đặc trưng lịch sử 7/30 ngày theo sân bay đi, sân bay đến, hãng, tuyến. Dự kiến tạo **một file đặc trưng dùng chung** cho cả hai mô hình (cách chia sẻ file sẽ thống nhất sau).
- Metric: ROC-AUC, AP, Brier + calibration, Recall/Precision tại top 10% (xem [flight_disruption.md](flight_disruption.md), mục 5).
