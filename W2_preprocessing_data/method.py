from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data") / "earthquake_1995-2023.csv"
OUTPUT_PATH = Path("data") / "monthly_features_step_1_8.csv"

REQUIRED_COLUMNS = [
    "date_time",
    "magnitude",
    "depth",
    "tsunami",
    "latitude",
    "longitude",
]


def step_1_load_main_dataset(input_path: Path = DATA_PATH) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    df = pd.read_csv(input_path)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    return df


def step_2_parse_datetime_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date_time"] = pd.to_datetime(
        df["date_time"],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )
    df = df.dropna(subset=["date_time"])
    df = df.sort_values("date_time").reset_index(drop=True)
    return df


def step_3_conservative_deduplication(df: pd.DataFrame) -> pd.DataFrame:
    dedup_cols = ["date_time", "magnitude", "latitude", "longitude", "depth"]
    return df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)


def step_4_monthly_resampling_and_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = ["magnitude", "depth", "tsunami"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    monthly_df = (
        df.set_index("date_time")
        .resample("MS")
        .agg(
            {
                "magnitude": ["count", "mean", "max", "min"],
                "depth": ["mean", "max", "min"],
                "tsunami": "sum",
            }
        )
    )

    monthly_df.columns = [
        "event_count",
        "mean_magnitude",
        "max_magnitude",
        "min_magnitude",
        "mean_depth",
        "max_depth",
        "min_depth",
        "tsunami_count",
    ]
    monthly_df.index.name = "month"
    return monthly_df


def step_5_missing_month_handling(monthly_df: pd.DataFrame) -> pd.DataFrame:
    monthly_df = monthly_df.copy().asfreq("MS")
    monthly_df["event_count"] = monthly_df["event_count"].fillna(0).astype(int)
    monthly_df["tsunami_count"] = monthly_df["tsunami_count"].fillna(0)
    return monthly_df


def step_6_missing_value_handling(monthly_df: pd.DataFrame) -> pd.DataFrame:
    monthly_df = monthly_df.copy()
    impute_cols = [
        "mean_magnitude",
        "max_magnitude",
        "min_magnitude",
        "mean_depth",
        "max_depth",
        "min_depth",
    ]

    for col in impute_cols:
        monthly_df[f"{col}_missing_flag"] = monthly_df[col].isna().astype(int)
        monthly_df[col] = monthly_df[col].ffill().bfill()

    return monthly_df


def step_7_outlier_detection_and_flagging(monthly_df: pd.DataFrame) -> pd.DataFrame:
    monthly_df = monthly_df.copy()

    roll_mean = monthly_df["event_count"].rolling(window=12, min_periods=6).mean()
    roll_std = monthly_df["event_count"].rolling(window=12, min_periods=6).std()
    safe_roll_std = roll_std.replace(0, np.nan)

    monthly_df["event_count_rolling_z"] = (
        monthly_df["event_count"] - roll_mean
    ) / safe_roll_std
    monthly_df["event_count_outlier_flag"] = (
        monthly_df["event_count_rolling_z"].abs() > 3
    ).astype(int)

    magnitude_q1 = monthly_df["max_magnitude"].quantile(0.25)
    magnitude_q3 = monthly_df["max_magnitude"].quantile(0.75)
    magnitude_iqr = magnitude_q3 - magnitude_q1
    monthly_df["max_magnitude_outlier_flag"] = (
        (monthly_df["max_magnitude"] < magnitude_q1 - 1.5 * magnitude_iqr)
        | (monthly_df["max_magnitude"] > magnitude_q3 + 1.5 * magnitude_iqr)
    ).astype(int)

    depth_q1 = monthly_df["mean_depth"].quantile(0.25)
    depth_q3 = monthly_df["mean_depth"].quantile(0.75)
    depth_iqr = depth_q3 - depth_q1
    monthly_df["mean_depth_outlier_flag"] = (
        (monthly_df["mean_depth"] < depth_q1 - 1.5 * depth_iqr)
        | (monthly_df["mean_depth"] > depth_q3 + 1.5 * depth_iqr)
    ).astype(int)

    return monthly_df


def step_8_feature_extraction(monthly_df: pd.DataFrame) -> pd.DataFrame:
    monthly_df = monthly_df.copy()

    monthly_df["event_count_lag_1"] = monthly_df["event_count"].shift(1)
    monthly_df["event_count_lag_3"] = monthly_df["event_count"].shift(3)
    monthly_df["event_count_lag_6"] = monthly_df["event_count"].shift(6)
    monthly_df["max_magnitude_lag_1"] = monthly_df["max_magnitude"].shift(1)
    monthly_df["mean_depth_lag_1"] = monthly_df["mean_depth"].shift(1)
    monthly_df["tsunami_count_lag_1"] = monthly_df["tsunami_count"].shift(1)

    monthly_df["event_count_roll_mean_3"] = (
        monthly_df["event_count"].rolling(3).mean()
    )
    monthly_df["event_count_roll_mean_6"] = (
        monthly_df["event_count"].rolling(6).mean()
    )
    monthly_df["event_count_roll_std_6"] = monthly_df["event_count"].rolling(6).std()
    monthly_df["event_count_roll_max_6"] = monthly_df["event_count"].rolling(6).max()
    monthly_df["max_magnitude_roll_max_6"] = (
        monthly_df["max_magnitude"].rolling(6).max()
    )
    monthly_df["mean_depth_roll_mean_6"] = monthly_df["mean_depth"].rolling(6).mean()
    monthly_df["tsunami_count_roll_sum_12"] = (
        monthly_df["tsunami_count"].rolling(12).sum()
    )

    monthly_df["event_count_delta_1"] = monthly_df["event_count"].diff(1)
    monthly_df["max_magnitude_delta_1"] = monthly_df["max_magnitude"].diff(1)
    monthly_df["mean_depth_delta_1"] = monthly_df["mean_depth"].diff(1)

    month_number = monthly_df.index.month
    monthly_df["month_sin"] = np.sin(2 * np.pi * month_number / 12)
    monthly_df["month_cos"] = np.cos(2 * np.pi * month_number / 12)

    return monthly_df


def build_step_1_to_8_features(input_path: Path = DATA_PATH) -> pd.DataFrame:
    df = step_1_load_main_dataset(input_path)
    df = step_2_parse_datetime_and_sort(df)
    df = step_3_conservative_deduplication(df)
    monthly_df = step_4_monthly_resampling_and_aggregation(df)
    monthly_df = step_5_missing_month_handling(monthly_df)
    monthly_df = step_6_missing_value_handling(monthly_df)
    monthly_df = step_7_outlier_detection_and_flagging(monthly_df)
    monthly_df = step_8_feature_extraction(monthly_df)
    return monthly_df


def main() -> None:
    monthly_features = build_step_1_to_8_features(DATA_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    monthly_features.to_csv(OUTPUT_PATH)

    print(f"Input: {DATA_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Monthly rows: {len(monthly_features)}")
    print(f"Feature columns: {len(monthly_features.columns)}")
    print(monthly_features.tail().to_string())


if __name__ == "__main__":
    main()
