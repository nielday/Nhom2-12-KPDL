# 🏨 Hotel Booking Cancellation Prediction

## Đề tài 12 – Khai phá dữ liệu (Data Mining) – HK2 2025-2026

### Mô tả
Dự án phân tích và dự đoán huỷ đặt phòng khách sạn sử dụng kỹ thuật khai phá dữ liệu, 
bao gồm: luật kết hợp, phân cụm, phân lớp, học bán giám sát, và chuỗi thời gian.

### Dataset
- **Nguồn**: [Hotel Booking Demand – Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **Kích thước**: 119,390 bookings × 32 features
- **Target**: `is_canceled` (0 = không huỷ, 1 = huỷ)
- **Thời gian**: July 2015 – August 2017
- **Khách sạn**: City Hotel & Resort Hotel (Portugal)

### Cài đặt
```bash
pip install -r requirements.txt
```

### Cấu trúc repo
```
KPDL/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── params.yaml           # Tham số: seed, split, hyperparams
├── data/
│   ├── raw/                   # Dữ liệu gốc
│   └── processed/             # Dữ liệu sau tiền xử lý
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocess_feature.ipynb
│   ├── 03_mining_or_clustering.ipynb
│   ├── 04_modeling.ipynb
│   ├── 04b_semi_supervised.ipynb
│   └── 05_evaluation_report.ipynb
├── src/
│   ├── data/                  # loader.py, cleaner.py
│   ├── features/              # builder.py
│   ├── mining/                # association.py, clustering.py
│   ├── models/                # supervised.py, semi_supervised.py, forecasting.py
│   ├── evaluation/            # metrics.py, report.py
│   └── visualization/         # plots.py
├── scripts/
│   ├── run_pipeline.py
│   └── run_papermill.py
├── app.py                     # Streamlit demo app
└── outputs/
    ├── figures/
    ├── tables/
    ├── models/
    └── reports/
```

### Chạy pipeline
```bash
# Cập nhật đường dẫn dữ liệu trong configs/params.yaml
python scripts/run_pipeline.py

# Hoặc chạy notebooks bằng papermill
python scripts/run_papermill.py
```

### Chạy demo app
```bash
streamlit run app.py
```

### Data Dictionary
| Cột | Mô tả |
|-----|-------|
| `hotel` | Loại khách sạn (City Hotel / Resort Hotel) |
| `is_canceled` | Booking có bị huỷ không (0/1) – **TARGET** |
| `lead_time` | Số ngày từ lúc đặt đến ngày đến |
| `arrival_date_year` | Năm đến |
| `arrival_date_month` | Tháng đến |
| `arrival_date_week_number` | Tuần trong năm |
| `arrival_date_day_of_month` | Ngày trong tháng |
| `stays_in_weekend_nights` | Số đêm cuối tuần |
| `stays_in_week_nights` | Số đêm trong tuần |
| `adults` | Số người lớn |
| `children` | Số trẻ em |
| `babies` | Số trẻ sơ sinh |
| `meal` | Loại bữa ăn (BB/FB/HB/SC/Undefined) |
| `country` | Quốc gia khách (ISO 3166-1 alpha-3) |
| `market_segment` | Phân khúc thị trường |
| `distribution_channel` | Kênh phân phối |
| `is_repeated_guest` | Khách quay lại (0/1) |
| `previous_cancellations` | Số lần huỷ trước đó |
| `previous_bookings_not_canceled` | Số booking không huỷ trước đó |
| `reserved_room_type` | Loại phòng đặt |
| `assigned_room_type` | Loại phòng được xếp |
| `booking_changes` | Số lần thay đổi booking |
| `deposit_type` | Loại đặt cọc (No Deposit/Non Refund/Refundable) |
| `agent` | ID đại lý |
| `company` | ID công ty |
| `days_in_waiting_list` | Số ngày trong danh sách chờ |
| `customer_type` | Loại khách (Transient/Contract/Group/Transient-Party) |
| `adr` | Average Daily Rate (giá trung bình/đêm) |
| `required_car_parking_spaces` | Số chỗ đậu xe yêu cầu |
| `total_of_special_requests` | Số yêu cầu đặc biệt |
| `reservation_status` | ⚠️ LEAKAGE – Trạng thái cuối cùng |
| `reservation_status_date` | ⚠️ LEAKAGE – Ngày trạng thái |
