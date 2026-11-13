# CLAUDE.md - Đồ án ML: Dự báo rủi ro gián đoạn chuyến bay (LightGBM vs Random Forest)

File này hướng dẫn Claude Code làm việc trong thư mục `Project/`. Ngôn ngữ trao đổi và comment trong code: **tiếng Việt**. Tên biến và hàm: tiếng Anh. Dùng `pathlib` cho đường dẫn (máy chạy Windows).

## 1. Bối cảnh

- Môn Machine Learning, nhóm 2 người, yêu cầu **2 model**: **LightGBM** (người dùng này, chạy **local**) và **Random Forest** (thành viên còn lại, chạy trên Kaggle).
- Bài toán: tại **T-24h** trước giờ khởi hành dự kiến, dự báo xác suất chuyến bay bị gián đoạn. Dữ liệu: BTS Reporting Carrier On-Time Performance, **13 tháng, 12/2024 đến 12/2025** (~540 nghìn chuyến/tháng). Tháng 12/2024 chỉ để khởi động lịch sử.
- Nếu có `docs/flight_disruption.md` / `README.md` thì đó là mô tả bài toán của nhóm. Nếu README và file này mâu thuẫn, **hỏi người dùng**, đừng tự chọn.
- Giai đoạn hiện tại: **bước chung** = nạp dữ liệu, tạo nhãn, chia tập theo thời gian, EDA. **Chưa tạo đặc trưng lịch sử, chưa huấn luyện, không tuning.**

## 2. Cấu trúc thư mục (theo template của nhóm, giữ nguyên, không đổi tên)

```
CLAUDE.md                 # ở thư mục gốc
docs/
  flight_disruption.md  # mô tả bài toán của nhóm
config/config.yaml        # đường dẫn, danh sách cột, mốc chia tập, seed
data/
  01-raw/                 # file BTS gốc (zip/csv theo tháng): KHÔNG sửa, KHÔNG commit
  02-preprocessed/        # flights.parquet (đã có nhãn + cột split): không commit
  03-features/            # đặc trưng (giai đoạn sau, chưa dùng)
  04-predictions/         # kết quả dự đoán (giai đoạn sau)
entrypoint/
  prepare_data.py         # chạy bước chung: nạp -> nhãn -> chia tập
  train.py, inference.py  # giai đoạn sau, chưa làm
models/                   # model đã train (giai đoạn sau)
notebooks/
  EDA.ipynb               # EDA bước chung (hình lưu notebooks/figures/)
  Baseline.ipynb          # giai đoạn sau
src/pipelines/
  utils.py                # load config, đường dẫn (pathlib)
  data_pipeline.py        # nạp từng tháng, tạo nhãn, gán split, kiểm tra
  feature_eng_pipeline.py, training_pipeline.py, inference_pipeline.py  # giai đoạn sau
tests/                    # test nhãn và chia tập (pytest)
```

Mọi hằng số (cột `usecols`, mốc chia, ngưỡng 60 phút, seed) đặt trong `config/config.yaml`, không hard-code. Mỗi file đặt đúng thư mục theo vai trò; không để code rải ngoài các thư mục này.

## 3. Cài đặt chung (phải giống thành viên kia)

**Cột được nạp (`usecols`)**: `FlightDate, Reporting_Airline, Flight_Number_Reporting_Airline, Origin, Dest, CRSDepTime, CRSArrTime, CRSElapsedTime, Distance, Cancelled, Diverted, ArrDelay, Tail_Number` (Tail_Number chỉ giữ lại, chưa dùng làm đặc trưng). Có thể thêm `DayOfWeek`, `Month` nếu cần. Bỏ cột rỗng `Unnamed: 109`. Ép kiểu nhỏ (float32, int16/int8, category cho chuỗi) vì một tháng đủ 110 cột tốn ~900 MB RAM.

**Cột CẤM làm đầu vào** (chỉ biết sau khi bay, gây rò rỉ): `DepTime, DepDelay, DepDelayMinutes, DepDel15, TaxiOut, WheelsOff, WheelsOn, TaxiIn, ArrTime, ArrDelayMinutes, ArrDel15, ActualElapsedTime, AirTime, CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay, CancellationCode`, các cột `Div*`. `ArrDelay`, `Cancelled`, `Diverted` chỉ dùng để tạo nhãn.

**Nhãn** (3 trường hợp, không điền thiếu bằng 0):

- `y = 1` nếu `Cancelled == 1` hoặc `Diverted == 1` hoặc `ArrDelay >= 60`.
- `y = 0` nếu không hủy, không chuyển hướng, `ArrDelay` hợp lệ và `< 60`.
- Còn lại: "chưa xác định" (`y` = NA), loại khỏi huấn luyện và đánh giá.
- Lưu thêm: `label_status` (`positive`/`negative`/`undetermined`), `y` kiểu `Int8` cho phép NA, cờ thành phần `is_cancelled`, `is_diverted`, `is_delay60` (trễ >= 60 và không hủy/chuyển hướng).

**Chia theo thời gian (theo `FlightDate`), KHÔNG random.** Cột `split` ∈ {`warmup`, `train`, `gap`, `val`, `test`}:

| split  | FlightDate                |
| ------ | ------------------------- |
| warmup | 2024-12-01 đến 2024-12-31 |
| train  | 2025-01-01 đến 2025-08-24 |
| gap    | 2025-08-25 đến 2025-08-31 |
| val    | 2025-09-01 đến 2025-10-24 |
| gap    | 2025-10-25 đến 2025-10-31 |
| test   | 2025-11-01 đến 2025-12-31 |

(Đề xuất, người dùng chưa chốt hẳn; **đọc mốc từ `config/config.yaml`, không hard-code ở nhiều nơi**.) Lưu **một** bản chia dùng chung, tạo một lần, không tự chia lại. Val dùng cho early stopping. Test chỉ dùng một lần ở cuối. Chuyến warmup/gap vẫn được dùng để tính lịch sử cho chuyến sau (giai đoạn sau), nhưng không dùng nhãn của chúng để huấn luyện/đánh giá.

**Kiểm tra bắt buộc sau khi chia:** các phần không giao nhau; ngày lớn nhất của phần trước < ngày nhỏ nhất của phần sau; in số dòng, tỉ lệ `y=1`, tỉ lệ từng thành phần, khoảng ngày của từng phần; mỗi tháng `ArrDelay` thiếu == hủy + chuyển hướng (nếu lệch thì báo).

**Phạm vi sân bay:** chưa chốt, quyết định sau EDA và **chỉ dựa trên giai đoạn train**. Chưa tự lọc.

**Mất cân bằng:** không xử lý (không class weight, không resample).

**Metric (giai đoạn sau):** ROC-AUC, AP, Brier + calibration, alert budget top 10%. Seed `42`. Ghi thời gian, RAM đỉnh, cấu hình máy.

**Lưu ý thời gian:** giờ dự kiến là số nguyên `hhmm` (cần đệm số 0); chưa có cột múi giờ (sẽ cần bảng múi giờ sân bay ở bước đặc trưng). Chưa xử lý ở bước này, chỉ thống kê giá trị `2400` ở mỗi tháng.

## 4. Môi trường local

- Windows, Python + `pandas`, `numpy`, `pyarrow`, `matplotlib`, `psutil`, `jupyter`; sau này `lightgbm`, `scikit-learn`. Ghi phiên bản vào `requirements.txt`; thành viên kia chạy Kaggle nên cần **cùng phiên bản** `lightgbm`, `scikit-learn`, `pandas`.
- RAM có hạn: nạp từng tháng với `usecols` + dtype nhỏ, giải phóng DataFrame không dùng, lưu parquet cho các bước sau. Ghi RAM đỉnh của mỗi bước nặng.
- Thời gian/RAM local **không so sánh trực tiếp** với Kaggle; luôn ghi kèm cấu hình máy.

## 5. Quy tắc làm việc

- **Không sửa, không xóa, không commit** file trong `data/`. Thêm `data/` vào `.gitignore`.
- **Không dùng nhãn của val/test** để fit bất cứ thứ gì (chọn sân bay, mã hóa, ngưỡng...).
- Mọi con số báo cáo phải đến từ lần chạy thật; chưa chạy thì ghi "chưa chạy". Không bịa hay làm tròn thay kết quả.
- Không tự thêm đặc trưng, tuning hay xử lý mất cân bằng khi chưa được yêu cầu. Cột dẫn xuất chỉ dùng cho vẽ biểu đồ EDA, không lưu vào file đầu vào mô hình.
- Với thay đổi ảnh hưởng cài đặt chung (mốc chia, định nghĩa nhãn, cột cấm, phạm vi), **dừng lại và hỏi**.
- Sau mỗi bước chính: in kiểm tra (shape, tỉ lệ y=1, RAM), rồi tóm tắt ngắn gồm file đã tạo và điều bất thường.
- Trước khi chạy lệnh tốn thời gian hoặc RAM nhiều, nói rõ sẽ làm gì và dự kiến mức tốn kém.

## 6. Chia sẻ với thành viên kia

Code và tài liệu bàn giao qua repo GitHub (private). Dữ liệu **không** nằm trong repo: `data/02-preprocessed/flights.parquet` và `prepare_data_report.json` (kèm các file dữ liệu sinh ra sau này) được đóng gói thành một file zip (vd `cs114_data_v1.zip`, đặt ngoài repo) và người dùng tự gửi trực tiếp. Zip giữ đường dẫn `data/02-preprocessed/...` để giải nén ở thư mục gốc repo, kèm `MD5SUMS.txt`. Mỗi lần tạo lại phải in số dòng, tỉ lệ y=1 từng phần và checksum (md5) để đối chiếu, tăng số phiên bản zip và cập nhật md5 trong `docs/README.md`.
