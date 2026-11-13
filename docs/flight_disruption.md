# Dự báo rủi ro gián đoạn chuyến bay bằng Machine Learning

Đồ án môn Machine Learning (nhóm 2 người). Hai mô hình chính: **LightGBM** và **Random Forest**.

## 1. Bài toán

**Mục tiêu:** từ lịch bay và lịch sử hoạt động, dự báo xác suất một chuyến bay bị **gián đoạn nghiêm trọng**: bị hủy, bị chuyển hướng, hoặc đến trễ từ 60 phút trở lên.

**Ứng dụng chính:** hỗ trợ người lập kế hoạch chuyến đi so sánh rủi ro giữa các phương án lịch bay (ví dụ các chuyến khác nhau trên cùng một tuyến). Đầu ra là xác suất để người dùng tự cân nhắc.

**Hướng mở rộng (ngoài phạm vi hiện tại):** hỗ trợ bố trí tổ bay, máy bay, nhân sự, hoặc dự đoán lỡ chuyến nối. Dữ liệu công khai hiện tại chưa đủ để trực tiếp thực hiện các chức năng này.

**Thời điểm dự báo:** một mốc duy nhất, **T-24h** (24 giờ trước giờ khởi hành dự kiến).

**Giả định và giới hạn (mô phỏng hồi cứu):**
- Dùng lịch bay trong bản ghi BTS làm đại diện cho lịch đã biết tại T-24h. Dataset không có bản chụp (snapshot) tại thời điểm đó, nên không kiểm chứng được lịch bay và trạng thái hủy đã được biết chính xác tại mốc này.
- Đây là mô phỏng trên dữ liệu đã có và **chưa tính độ trễ phát hành dữ liệu của BTS** (dữ liệu mỗi tháng chỉ được công bố sau một khoảng trễ; ví dụ tháng 8/2026 phát hành ngày 19/10/2026). Vì vậy kết quả chưa chứng minh được mô hình dùng được trực tiếp trong thực tế.

**Lý do chọn ngưỡng 60 phút:** đây là **định nghĩa vận hành của đồ án**, chọn để tập trung vào gián đoạn đáng kể; không phải ngưỡng trễ 15 phút thường dùng trong thống kê BTS. Nếu còn thời gian, có thể kiểm tra độ nhạy với các ngưỡng khác (ví dụ 30, 90, 120 phút) như một phần mở rộng tùy chọn.

## 2. Dữ liệu

**Nguồn:** [U.S. Bureau of Transportation Statistics (BTS) - Airline On-Time Performance](https://transtats.bts.gov/ONTIME/). Theo trang nguồn, dữ liệu có từ 01/1995 đến 08/2026; giờ ghi theo giờ địa phương, định dạng 24 giờ.

**Phạm vi:** chuyến bay nội địa Mỹ của các hãng báo cáo cho BTS. Mỗi dòng là **một chuyến bay**. **Khoảng thời gian: 13 tháng, từ 12/2024 đến 12/2025** (khoảng 540 nghìn chuyến mỗi tháng). Tháng 12/2024 chỉ dùng để khởi động cửa sổ lịch sử, không thuộc train, validation hay test. Phạm vi sân bay sẽ chốt sau EDA (mục 7).

**Các nhóm cột chính** (tên cột theo bản BTS thường gặp, cần đối chiếu với file thực tế khi tải):

| Nhóm | Cột ví dụ | Vai trò |
|---|---|---|
| Lịch bay | `FlightDate`, hãng bay, số hiệu chuyến, `Origin`, `Dest`, `CRSDepTime`, `CRSArrTime`, thời gian bay dự kiến, `Distance` | **Đầu vào** |
| Số đuôi máy bay | `Tail_Number` | **Không dùng ở phiên bản đầu**: chưa có bằng chứng số đuôi trong bản ghi cuối cùng đã được phân công và biết trước tại T-24h |
| Kết quả thực tế | `DepTime`, `DepDelay`, `TaxiOut`, `WheelsOff`, `WheelsOn`, `TaxiIn`, `ArrTime`, `AirTime`, thời gian bay thực tế | **Cấm dùng làm đầu vào** (chỉ biết sau khi bay) |
| Trạng thái chuyến bay | `Cancelled`, `Diverted`, `ArrDelay` / `ArrDelayMinutes`, `CancellationCode` | Chỉ dùng để **tạo nhãn** |
| Nguyên nhân trễ | Các cột nguyên nhân trễ, gồm `WeatherDelay` | **Cấm dùng làm đầu vào** (thông tin nguyên nhân ghi nhận sau sự kiện) |

**Hạn chế đã biết**
- Bảng dữ liệu đang dùng **không có số đo hoặc dự báo khí tượng trực tiếp**. `WeatherDelay` là nguyên nhân ghi nhận sau sự kiện, không phải thông tin thời tiết biết trước. Có thể ghép thêm dữ liệu thời tiết bên ngoài như một mở rộng tùy chọn.
- Với chuyến hủy hoặc chuyển hướng, các trường kết quả thông thường có thể thiếu hoặc không áp dụng; xử lý theo trạng thái chuyến (mục 4). Một số chuyến chuyển hướng vẫn đến được đích dự kiến.
- Số liệu khảo sát ban đầu (**chỉ tháng 1/2025**: 539.747 chuyến, 329 sân bay, 14 hãng): hủy 3,02%, chuyển hướng 0,22%, trễ từ 60 phút 6,16%, nhãn 1 tổng 9,40%; không có dòng "chưa xác định"; `ArrDelay` thiếu đúng bằng tỉ lệ hủy cộng chuyển hướng (3,24%). Tháng 1 là mùa đông nên các tỉ lệ này chưa đại diện cho cả năm; sẽ cập nhật sau EDA toàn bộ 13 tháng.
- File BTS tải về có 110 cột, trong đó có một cột rỗng `Unnamed: 109` (do dấu phẩy thừa cuối dòng) cần bỏ khi nạp. Một tháng nạp đủ cột tốn khoảng 900 MB RAM, nên chỉ nạp các cột cần dùng.

## 3. Input

Mỗi mẫu là **một chuyến bay**, dự báo tại T-24h. Đầu vào chỉ gồm thông tin đã biết **trước** mốc đó.

**Lịch bay**
- Thời gian: ngày, thứ trong tuần, tháng, giờ khởi hành dự kiến, giờ đến dự kiến.
- Chuyến bay: hãng bay, số hiệu chuyến, sân bay đi, sân bay đến, khoảng cách, thời gian bay dự kiến.

**Lịch sử hoạt động** (tính riêng cho từng chuyến, tại mốc T-24h của chính chuyến đó)
- Tỉ lệ gián đoạn của sân bay đi, sân bay đến, hãng bay và tuyến bay trên các cửa sổ lùi 7 ngày và 30 ngày kết thúc tại T-24h.
- Chỉ dùng các chuyến **đã có kết quả trước T-24h**, tức chuyến đã hạ cánh hoặc đã bị hủy trước thời điểm đó.

**Quy tắc chống rò rỉ**
- Không dùng các cột thuộc nhóm "Kết quả thực tế" và "Nguyên nhân trễ".
- `Cancelled`, `Diverted`, `ArrDelay` chỉ dùng để tạo nhãn.
- Chia dữ liệu **theo thời gian**, không chia ngẫu nhiên.

## 4. Output

**Nhãn nhị phân** của mỗi chuyến bay, với $C_i$ là hủy, $D_i$ là chuyển hướng, $\Delta_i$ là số phút đến trễ:

$$
y_i =
\begin{cases}
1, & C_i = 1 \ \lor\ D_i = 1 \ \lor\ \Delta_i \ge 60,\\
0, & C_i = 0 \ \land\ D_i = 0 \ \land\ \Delta_i \text{ hợp lệ} \ \land\ \Delta_i < 60,\\
\text{chưa xác định}, & \text{các trường hợp còn lại.}
\end{cases}
$$

| Trường hợp | Xử lý |
|---|---|
| Xác định được hủy, chuyển hướng, hoặc đến trễ từ 60 phút | Nhãn 1 |
| Không hủy, không chuyển hướng, độ trễ hợp lệ dưới 60 phút (kể cả đến sớm) | Nhãn 0 |
| Không đủ dữ liệu để xác định | Đánh dấu "chưa xác định", kiểm tra riêng |

**Không điền độ trễ thiếu bằng 0.** Báo cáo số mẫu chưa xác định hoặc bị loại.

**Đầu ra của mô hình:** xác suất gián đoạn $\hat{p}_i \in [0, 1]$ cho từng chuyến bay tại T-24h.

## 5. Thiết lập đánh giá

- **ROC-AUC.**
- **AP (Average Precision)** làm thước đo chính cho lớp dương. Nếu dùng `average_precision_score` thì gọi đúng tên là AP, không gọi là PR-AUC.
- **Brier score** và **biểu đồ calibration** (độ tin cậy của xác suất).
- **Ngân sách cảnh báo:** top 10% chuyến có điểm rủi ro cao nhất (mặc định, xác nhận lại sau EDA). Báo cáo Recall và Precision tại ngân sách này.
- **Kịch bản chi phí giả định:** nhóm chưa có số liệu chi phí vận hành thực tế. Dùng một vài kịch bản với tỉ lệ chi phí bỏ sót : báo nhầm (ví dụ 1:1, 3:1, 10:1); ngưỡng cảnh báo được chọn **trên validation** cho từng kịch bản.
- Báo cáo riêng cho từng thành phần của lớp dương (hủy, chuyển hướng, trễ từ 60 phút).
- Calibration, chọn ngưỡng và tuning chỉ dùng train và validation; **test giữ riêng cho đánh giá cuối cùng**.

## 6. Baseline và thí nghiệm

- **Baseline đơn giản:** tỉ lệ gián đoạn lịch sử có làm trơn (ví dụ theo tuyến hoặc sân bay, làm trơn về tỉ lệ chung).
- **Hai mô hình chính:** LightGBM và Random Forest, so sánh trên **cùng tập mẫu, cùng nhãn, cùng cách chia thời gian**.
- **Thí nghiệm chính:**
  1. Chỉ lịch bay → thêm lịch sử hoạt động.
  2. Trước → sau calibration.
- **Mở rộng tùy thời gian** (không bắt buộc): độ nhạy với ngưỡng 30, 90, 120 phút.

## 7. Quy tắc thời gian và phạm vi

- **Ngày giờ và T-24h:** ghép ngày bay và giờ khởi hành dự kiến với múi giờ của sân bay đi, đổi sang UTC, rồi trừ đúng 24 giờ. Các trường giờ của BTS là giờ địa phương và bảng tải về không có cột múi giờ, nên cần bảng múi giờ sân bay từ nguồn ngoài.
- **Chuyến qua ngày:** không chỉ so sánh hai giá trị `hhmm`, vì giờ đi và giờ đến thuộc hai múi giờ khác nhau. Ngày đến xác định từ thời điểm khởi hành (UTC) cộng thời gian bay dự kiến, rồi đổi sang giờ địa phương của sân bay đến.
- **Trường hợp đặc biệt:** giờ dự kiến là số nguyên `hhmm` (ví dụ `5` là 00:05, `659` là 06:59) nên phải đệm số 0 trước khi chuyển đổi. Tháng 1/2025 không có giá trị `2400` ở `CRSDepTime` và `CRSArrTime`; kiểm tra lại với các tháng còn lại. Các trường hợp đổi giờ mùa hè (giờ không tồn tại hoặc bị lặp) xử lý riêng; thống kê số dòng bị ảnh hưởng.
- **Chia theo thời gian** (theo `FlightDate`, đề xuất, chốt sau EDA):

  | Phần | Ngày bay |
  |---|---|
  | Khởi động (chỉ để tính lịch sử) | 2024-12-01 đến 2024-12-31 |
  | **Train** | 2025-01-01 đến 2025-08-24 |
  | Khoảng đệm 7 ngày (bỏ) | 2025-08-25 đến 2025-08-31 |
  | **Validation** | 2025-09-01 đến 2025-10-24 |
  | Khoảng đệm 7 ngày (bỏ) | 2025-10-25 đến 2025-10-31 |
  | **Test** | 2025-11-01 đến 2025-12-31 |

  Chuyến bay ở phần khởi động và khoảng đệm vẫn được dùng để tính lịch sử cho các chuyến sau, nhưng không dùng nhãn của chúng để huấn luyện hay đánh giá. Test rơi vào mùa lễ và đầu đông, khác mùa với phần lớn train; cần nêu rõ trong báo cáo. Mô hình được cố định trong suốt giai đoạn test.
- **Chốt sau EDA:** phạm vi sân bay (nếu chọn theo số chuyến thì **chỉ dùng giai đoạn train** để lựa chọn) và việc điều chỉnh các mốc chia trên nếu cần.
