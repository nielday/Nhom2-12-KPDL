"""
Data loader module.

Handles reading raw data, schema validation, and data dictionary.
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "configs/params.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


# Expected schema for hotel_bookings.csv
EXPECTED_COLUMNS = [
    "hotel", "is_canceled", "lead_time", "arrival_date_year",
    "arrival_date_month", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies",
    "meal", "country", "market_segment", "distribution_channel",
    "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "reserved_room_type",
    "assigned_room_type", "booking_changes", "deposit_type",
    "agent", "company", "days_in_waiting_list", "customer_type",
    "adr", "required_car_parking_spaces",
    "total_of_special_requests", "reservation_status",
    "reservation_status_date",
]


def load_raw_data(
    path: str = None,
    config_path: str = "configs/params.yaml",
) -> pd.DataFrame:
    """
    Load raw hotel bookings data from CSV.

    Parameters
    ----------
    path : str, optional
        Path to CSV file. If None, reads from config.
    config_path : str
        Path to YAML config file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with validated schema.

    Raises
    ------
    FileNotFoundError
        If data file does not exist.
    ValueError
        If schema validation fails.
    """
    if path is None:
        config = load_config(config_path)
        path = config["data"]["raw_path"]

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path.absolute()}"
        )

    df = pd.read_csv(data_path)

    # Validate schema
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing columns in dataset: {missing_cols}"
        )

    print(f"[Loader] Loaded {df.shape[0]:,} rows × "
          f"{df.shape[1]} columns from {path}")
    return df


def get_data_dictionary() -> Dict[str, str]:
    """
    Return data dictionary describing each column.

    Returns
    -------
    dict
        Column name -> description mapping.
    """
    return {
        "hotel": (
            "Loại khách sạn: City Hotel hoặc Resort Hotel"
        ),
        "is_canceled": (
            "Booking bị huỷ hay không (0=không, 1=huỷ) "
            "– TARGET variable"
        ),
        "lead_time": (
            "Số ngày giữa ngày đặt phòng và ngày đến "
            "(0 = đặt ngay ngày đến)"
        ),
        "arrival_date_year": "Năm khách đến (2015-2017)",
        "arrival_date_month": "Tháng khách đến (January-December)",
        "arrival_date_week_number": "Tuần trong năm (1-53)",
        "arrival_date_day_of_month": "Ngày trong tháng (1-31)",
        "stays_in_weekend_nights": (
            "Số đêm cuối tuần (Sat/Sun) khách ở"
        ),
        "stays_in_week_nights": (
            "Số đêm trong tuần (Mon-Fri) khách ở"
        ),
        "adults": "Số người lớn",
        "children": "Số trẻ em",
        "babies": "Số trẻ sơ sinh",
        "meal": (
            "Loại bữa ăn đặt: BB (Bed & Breakfast), "
            "HB (Half Board), FB (Full Board), "
            "SC (Self Catering), Undefined"
        ),
        "country": (
            "Quốc gia khách hàng (ISO 3166-1 alpha-3)"
        ),
        "market_segment": (
            "Phân khúc thị trường: Online TA, Offline TA/TO, "
            "Groups, Direct, Corporate, Complementary, Aviation"
        ),
        "distribution_channel": (
            "Kênh phân phối: Direct, Corporate, TA/TO, "
            "GDS, Undefined"
        ),
        "is_repeated_guest": "Khách quay lại hay không (0/1)",
        "previous_cancellations": (
            "Số lần huỷ booking trước đó của khách"
        ),
        "previous_bookings_not_canceled": (
            "Số booking trước đó không bị huỷ"
        ),
        "reserved_room_type": "Loại phòng khách đặt (A-L)",
        "assigned_room_type": (
            "Loại phòng thực tế được xếp (A-L)"
        ),
        "booking_changes": "Số lần thay đổi booking",
        "deposit_type": (
            "Loại đặt cọc: No Deposit, Non Refund, Refundable"
        ),
        "agent": "ID đại lý đặt phòng (NaN = không qua đại lý)",
        "company": (
            "ID công ty đặt phòng (NaN = không qua công ty)"
        ),
        "days_in_waiting_list": (
            "Số ngày booking nằm trong danh sách chờ"
        ),
        "customer_type": (
            "Loại khách: Transient, Contract, "
            "Transient-Party, Group"
        ),
        "adr": "Average Daily Rate – giá trung bình mỗi đêm (€)",
        "required_car_parking_spaces": (
            "Số chỗ đậu xe khách yêu cầu"
        ),
        "total_of_special_requests": (
            "Số yêu cầu đặc biệt (phòng tầng cao, giường phụ…)"
        ),
        "reservation_status": (
            "⚠️ LEAKAGE – Trạng thái cuối cùng: "
            "Check-Out / Canceled / No-Show"
        ),
        "reservation_status_date": (
            "⚠️ LEAKAGE – Ngày cập nhật trạng thái cuối cùng"
        ),
    }
