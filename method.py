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

def s9_create_target(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    df = df.copy()
    df[target_col] = df["event_count"].shift(-1)
    df["target_month"] = df["month"] + pd.offsets.MonthBegin(1)

    shift_check = df[target_col].iloc[:-1].to_numpy() == df["event_count"].iloc[1:].to_numpy()
    if not shift_check.all():
        raise AssertionError("Next-month target shift is incorrect.")

    return df.dropna(subset=[*feature_cols, target_col]).reset_index(drop=True)

def s10_chronological_split(
    target_df: pd.DataFrame,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    test_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = target_df.loc[target_df["target_month"] <= train_end].copy()
    val_df = target_df.loc[target_df["target_month"].between(val_start, val_end)].copy()
    test_df = target_df.loc[target_df["target_month"] >= test_start].copy()

    if len(train_df) + len(val_df) + len(test_df) != len(target_df):
        raise AssertionError("Chronological splits do not cover the complete target dataset.")

    return train_df, val_df, test_df

def s11_fit_scaler(
    train_df: pd.DataFrame, feature_cols: list[str]
) -> tuple[RobustScaler, pd.DataFrame]:
    scaler = RobustScaler()
    scaler.fit(train_df[feature_cols])

    train_scaled_df = train_df.copy()
    train_scaled_df[feature_cols] = scaler.transform(train_df[feature_cols])
    if not np.isfinite(train_scaled_df[feature_cols].to_numpy()).all():
        raise AssertionError("Training features contain non-finite values after scaling.")

    return scaler, train_scaled_df

def s12_transform_splits(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler: RobustScaler,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_scaled_df = val_df.copy()
    test_scaled_df = test_df.copy()
    val_scaled_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_scaled_df[feature_cols] = scaler.transform(test_df[feature_cols])

    for split_name, frame in [("validation", val_scaled_df), ("test", test_scaled_df)]:
        if not np.isfinite(frame[feature_cols].to_numpy()).all():
            raise AssertionError(f"{split_name} features contain non-finite values after scaling.")

    return val_scaled_df, test_scaled_df

WINDOW_SIZE = 12
def create_lstm_windows(
    timeline_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    window_size: int = 12,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X, y, metadata = [], [], []

    for end_index in range(window_size - 1, len(timeline_df)):
        start_index = end_index - window_size + 1
        window_df = timeline_df.iloc[start_index : end_index + 1]
        expected_window_months = pd.date_range(
            window_df["month"].iloc[0],
            window_df["month"].iloc[-1],
            freq="MS",
        )
        if len(expected_window_months) != window_size or not np.array_equal(
            window_df["month"].to_numpy(), expected_window_months.to_numpy()
        ):
            continue

        sample_row = timeline_df.iloc[end_index]
        X.append(window_df[feature_columns].to_numpy(dtype=np.float32))
        y.append([np.float32(sample_row[target_column])])
        metadata.append(
            {
                "input_start_month": window_df["month"].iloc[0],
                "input_end_month": window_df["month"].iloc[-1],
                "target_month": sample_row["target_month"],
            }
        )

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), pd.DataFrame(metadata)

SEED = 42
np.random.seed(SEED)
LSTM_UNITS    = 32
DROPOUT_RATE  = 0.2
DENSE_UNITS   = 16
LEARNING_RATE = 0.001
BATCH_SIZE    = 16
EPOCHS        = 20
PATIENCE      = 10

def s14_build_and_train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    window_size: int,
    n_features: int,
    lstm_units: int,
    dropout_rate: float,
    dense_units: int,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    patience: int,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window_size, n_features)),
            tf.keras.layers.LSTM(lstm_units),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(dense_units, activation="relu"),
            tf.keras.layers.Dense(1),
        ],
        name="earthquake_count_lstm",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    model.summary()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=2,
    )
    return model, history

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
