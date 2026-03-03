# 🏨 Hotel Booking Cancellation Prediction

## Đề tài 12 – Khai phá dữ liệu (Data Mining) – HK2 2025-2026

### Nhóm 2

### Mô tả
Dự án phân tích và dự đoán huỷ đặt phòng khách sạn sử dụng kỹ thuật khai phá dữ liệu,
bao gồm: luật kết hợp (Apriori), phân cụm (KMeans/DBSCAN), phân lớp (LogReg, DT, RF, XGBoost),
học bán giám sát (Self-Training), và dự báo chuỗi thời gian (ARIMA, Holt-Winters).

### Dataset
- **Nguồn**: [Hotel Booking Demand – Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **Kích thước**: 119,390 bookings × 32 features
- **Target**: `is_canceled` (0 = không huỷ, 1 = huỷ)
- **Thời gian**: July 2015 – August 2017
- **Khách sạn**: City Hotel & Resort Hotel (Portugal)

---

## Kết quả chính

### Classification (Best: XGBoost)
| Model | Accuracy | F1 | PR-AUC | ROC-AUC | Train Time |
|---|---|---|---|---|---|
| **XGBoost** 🏆 | 85.3% | **0.716** | **0.809** | **0.917** | 0.71s |
| RandomForest | 77.2% | 0.673 | 0.740 | 0.885 | 1.46s |
| LogisticRegression | 72.7% | 0.604 | 0.620 | 0.812 | 8.68s |
| DecisionTree | 70.3% | 0.614 | 0.524 | 0.805 | 0.27s |

### Top Feature Importance
1. `room_mismatch` (16.6%) — phòng xếp khác phòng đặt
2. `required_car_parking_spaces` (11.4%)
3. `market_segment_Online TA` (10.7%)
4. `is_local` (8.1%)
5. `deposit_type_Non Refund` (6.0%)

### Time Series (Best: Moving Average)
| Method | MAE | RMSE | sMAPE |
|---|---|---|---|
| **Moving Average (3)** | 0.056 | **0.059** | **18.2%** |
| Naive | 0.065 | 0.070 | 21.3% |
| Holt-Winters | 0.069 | 0.076 | 22.9% |
| ARIMA(1,1,1) | 0.073 | 0.080 | 24.4% |

---

## Cài đặt
```bash
pip install -r requirements.txt
```

## Cấu trúc repo
```
KPDL/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── params.yaml              # Tham số: seed, split, hyperparams
├── data/
│   ├── raw/                     # Dữ liệu gốc (hotel_bookings.csv)
│   └── processed/               # Dữ liệu sau tiền xử lý
├── notebooks/
│   ├── 01_eda.ipynb             # EDA + data dictionary + leakage check
│   ├── 02_preprocess_feature.ipynb  # Cleaning + 11 features mới
│   ├── 03_mining_or_clustering.ipynb # Apriori + KMeans/DBSCAN
│   ├── 04_modeling.ipynb        # 4 models + confusion matrix + ROC/PR
│   ├── 04b_semi_supervised.ipynb # Self-Training + learning curves
│   └── 05_evaluation_report.ipynb # Time series + insights + tổng kết
├── src/
│   ├── data/                    # loader.py, cleaner.py
│   ├── features/                # builder.py (11 features)
│   ├── mining/                  # association.py, clustering.py
│   ├── models/                  # supervised.py, semi_supervised.py, forecasting.py
│   ├── evaluation/              # metrics.py, report.py
│   └── visualization/           # plots.py (18 hàm vẽ)
├── scripts/
│   ├── run_pipeline.py          # Chạy pipeline end-to-end
│   └── run_papermill.py         # Chạy notebooks tự động
├── app.py                       # Streamlit demo app (điểm thưởng)
└── outputs/
    ├── figures/                 # 17 biểu đồ PNG
    ├── tables/                  # 12 files CSV + JSON
    └── models/                  # best_model.joblib
```

## Cách chạy

### 1. Pipeline tự động
```bash
python scripts/run_pipeline.py
```

### 2. Notebooks (xem kết quả chi tiết)
```bash
jupyter notebook notebooks/
# Chạy theo thứ tự: 01 → 02 → 03 → 04 → 04b → 05
```

### 3. Streamlit demo app
```bash
streamlit run app.py
```

---

## Data Dictionary
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

## ⚠️ Data Leakage
Cột `reservation_status` và `reservation_status_date` chứa thông tin SAU khi booking kết thúc → **phải loại bỏ** trước khi train model.
