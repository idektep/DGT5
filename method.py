# Forecasting
def split_metadata(split_name: str, expected_rows: int) -> pd.DataFrame:
    split_df = metadata.loc[metadata["split"] == split_name].copy().reset_index(drop=True)
    if len(split_df) != expected_rows:
        raise AssertionError(
            f"{split_name} metadata rows ({len(split_df)}) do not match target rows ({expected_rows})."
        )
    return split_df


def make_forecast_frame(split_name: str, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise AssertionError(f"{split_name} arrays contain NaN or infinite values.")

    split_df = split_metadata(split_name, len(y))
    y_true = y.reshape(-1).astype(float)
    y_pred_raw = model.predict(X, verbose=0).reshape(-1).astype(float)
    y_pred = np.clip(y_pred_raw, 0.0, None)

    forecast_df = split_df[["input_start_month", "input_end_month", "target_month"]].copy()
    forecast_df["actual_event_count"] = y_true
    forecast_df["predicted_event_count_raw"] = y_pred_raw
    forecast_df["predicted_event_count"] = y_pred
    forecast_df["forecast_error"] = y_pred - y_true
    forecast_df["absolute_error"] = np.abs(forecast_df["forecast_error"])
    forecast_df["squared_error"] = forecast_df["forecast_error"] ** 2
    return forecast_df


train_forecast_df = make_forecast_frame("train", X_train, y_train)
val_forecast_df = make_forecast_frame("validation", X_val, y_val)
test_forecast_df = make_forecast_frame("test", X_test, y_test)

print("Validation forecast preview")
display(val_forecast_df.head())
display(val_forecast_df.tail())

print("Test forecast preview")
display(test_forecast_df.head())
display(test_forecast_df.tail())

# Evaluate
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    values = np.where(denominator == 0, 0.0, np.abs(y_pred - y_true) / denominator)
    return float(np.mean(values) * 100.0)


def evaluate_forecast(split_name: str, forecast_df: pd.DataFrame) -> dict[str, float | str | int]:
    y_true = forecast_df["actual_event_count"].to_numpy(dtype=float)
    y_pred = forecast_df["predicted_event_count"].to_numpy(dtype=float)
    error = y_pred - y_true
    return {
        "split": split_name,
        "samples": len(forecast_df),
        "first_target_month": forecast_df["target_month"].min().date().isoformat(),
        "last_target_month": forecast_df["target_month"].max().date().isoformat(),
        "actual_mean": float(np.mean(y_true)),
        "predicted_mean": float(np.mean(y_pred)),
        "bias_mean_error": float(np.mean(error)),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "sMAPE_percent": smape(y_true, y_pred),
    }


metrics_df = pd.DataFrame(
    [
        evaluate_forecast("train", train_forecast_df),
        evaluate_forecast("validation", val_forecast_df),
        evaluate_forecast("test", test_forecast_df),
    ]
)

display(metrics_df)

# Overfitting 
def metric_value(split_name: str, metric_name: str) -> float:
    return float(metrics_df.loc[metrics_df["split"] == split_name, metric_name].iloc[0])


def ratio_status(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "not available"
    if ratio <= 1.25:
        return "small gap"
    if ratio <= 1.75:
        return "moderate gap"
    return "large gap; inspect overfitting risk"


train_mae = metric_value("train", "MAE")
val_mae = metric_value("validation", "MAE")
test_mae = metric_value("test", "MAE")
val_train_ratio = val_mae / train_mae if train_mae else np.nan
test_train_ratio = test_mae / train_mae if train_mae else np.nan

if {"loss", "val_loss"}.issubset(history_df.columns):
    best_val_idx = int(history_df["val_loss"].idxmin())
    best_epoch = int(history_df.loc[best_val_idx, "epoch"])
    best_val_loss = float(history_df.loc[best_val_idx, "val_loss"])
    final_train_loss = float(history_df["loss"].iloc[-1])
    final_val_loss = float(history_df["val_loss"].iloc[-1])
    final_val_to_train_loss_ratio = final_val_loss / final_train_loss if final_train_loss else np.nan
    final_val_vs_best_val_percent = ((final_val_loss - best_val_loss) / best_val_loss * 100.0) if best_val_loss else np.nan
else:
    best_epoch = np.nan
    best_val_loss = np.nan
    final_train_loss = np.nan
    final_val_loss = np.nan
    final_val_to_train_loss_ratio = np.nan
    final_val_vs_best_val_percent = np.nan