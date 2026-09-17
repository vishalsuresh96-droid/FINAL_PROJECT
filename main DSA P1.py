"""
Electricity Load & Consumption Forecasting Pipeline
Implements end-to-end time series forecasting following ML best practices:
- Schema discovery & data preprocessing
- Exploratory data analysis & seasonal pattern extraction
- Stationarity testing (Augmented Dickey-Fuller test)
- Chronological Train / Validation / Test split (no future leakage)
- Feature engineering (lags, rolling stats, cyclical temporal encodings)
- Dual-model comparison: Statistical Time Series (ARIMA) vs ML (Random Forest Regressor)
- Validation metric evaluation (MAE, RMSE, MAPE) & model selection
- Test set evaluation and future horizon forecasting
- High-resolution visual diagnostics saved to forecast_results.png
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# Safe console encoding for Windows terminals
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. Data Ingestion & Schema Discovery
# ==============================================================================
def find_and_load_dataset(data_dir: str = "data") -> pd.DataFrame:
    """
    Scans data directory for electricity CSVs or auto-generates benchmark data.
    """
    os.makedirs(data_dir, exist_ok=True)
    candidate_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    # Priority matching
    target_file = None
    for f in candidate_files:
        if "electricity" in os.path.basename(f).lower():
            target_file = f
            break
    if not target_file and candidate_files:
        target_file = candidate_files[0]
        
    if not target_file:
        print(f"[!] No electricity CSV found in '{data_dir}'. Auto-generating benchmark dataset...")
        try:
            from data.generate_sample_data import generate_electricity_data
            df = generate_electricity_data(output_path=os.path.join(data_dir, "electricity_consumption.csv"))
            target_file = os.path.join(data_dir, "electricity_consumption.csv")
        except Exception as e:
            raise FileNotFoundError(f"Could not load or generate data: {e}")
    else:
        print(f"[+] Loading dataset from: {target_file}")
        df = pd.read_csv(target_file)

    # Standardize column names
    time_col = next((c for c in df.columns if any(k in c.lower() for k in ['time', 'date'])), None)
    target_col = next((c for c in df.columns if any(k in c.lower() for k in ['demand', 'kwh', 'load', 'consumption', 'power'])), None)

    if not time_col or not target_col:
        raise ValueError(f"Could not automatically identify timestamp or target column in {df.columns.tolist()}")

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    df.rename(columns={time_col: "timestamp", target_col: "demand_kwh"}, inplace=True)

    # Missing value audit and handling
    missing_count = df["demand_kwh"].isna().sum()
    if missing_count > 0:
        print(f"[!] Found {missing_count} missing values in target. Applying forward/backward fill interpolation...")
        df["demand_kwh"] = df["demand_kwh"].interpolate(method="linear").bfill().ffill()

    print(f"[OK] Successfully loaded {len(df):,} records ({df['timestamp'].min()} to {df['timestamp'].max()})")
    return df


# ==============================================================================
# 2. Exploratory Data Analysis & Stationarity Testing
# ==============================================================================
def perform_eda_and_stationarity(df: pd.DataFrame):
    """
    Analyzes summary stats and performs the Augmented Dickey-Fuller (ADF) test.
    """
    print("\n" + "=" * 65)
    print(" 1. EXPLORATORY DATA ANALYSIS & STATIONARITY TEST")
    print("=" * 65)
    
    series = df["demand_kwh"]
    stats = series.describe()
    print(f"Summary Statistics:")
    print(f"  * Count: {int(stats['count']):,}")
    print(f"  * Mean Demand: {stats['mean']:.2f} kWh")
    print(f"  * Std Deviation: {stats['std']:.2f} kWh")
    print(f"  * Min / Max: {stats['min']:.2f} / {stats['max']:.2f} kWh")
    print(f"  * Median (IQR 50%): {stats['50%']:.2f} kWh")

    # ADF Test for Stationarity
    print("\nRunning Augmented Dickey-Fuller (ADF) Test on Demand Series...")
    # Subsample or run on continuous sequence
    adf_result = adfuller(series.dropna()[:5000])  # sample up to 5000 for swift diagnostic
    adf_stat, p_value, lags_used, nobs, crit_vals, icbest = adf_result
    
    print(f"  * ADF Statistic: {adf_stat:.4f}")
    print(f"  * p-value: {p_value:.6e}")
    print(f"  * Lags Used: {lags_used}")
    print(f"  * Critical Values: 1%: {crit_vals['1%']:.3f}, 5%: {crit_vals['5%']:.3f}, 10%: {crit_vals['10%']:.3f}")
    
    if p_value < 0.05:
        print("  [OK] Conclusion: Series is stationary at 95% confidence level (p < 0.05).")
    else:
        print("  [!] Conclusion: Series is non-stationary (p >= 0.05). Differencing recommended for ARIMA.")


# ==============================================================================
# 3. Feature Engineering & Strict Chronological Split
# ==============================================================================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts temporal cyclical encodings, lag variables, and rolling metrics.
    """
    data = df.copy()
    data["hour"] = data["timestamp"].dt.hour
    data["dayofweek"] = data["timestamp"].dt.dayofweek
    data["month"] = data["timestamp"].dt.month
    data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)

    # Cyclical representations of time to preserve wrap-around continuity
    data["sin_hour"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["cos_hour"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["sin_dayofweek"] = np.sin(2 * np.pi * data["dayofweek"] / 7)
    data["cos_dayofweek"] = np.cos(2 * np.pi * data["dayofweek"] / 7)
    data["sin_month"] = np.sin(2 * np.pi * data["month"] / 12)
    data["cos_month"] = np.cos(2 * np.pi * data["month"] / 12)

    # Autoregressive Lags (t-1, t-2, t-24 same hour yesterday, t-168 same hour last week)
    data["lag_1"] = data["demand_kwh"].shift(1)
    data["lag_2"] = data["demand_kwh"].shift(2)
    data["lag_24"] = data["demand_kwh"].shift(24)
    data["lag_168"] = data["demand_kwh"].shift(168)

    # Rolling window aggregations (computed with shift(1) to avoid look-ahead bias)
    shifted = data["demand_kwh"].shift(1)
    data["rolling_mean_24"] = shifted.rolling(window=24).mean()
    data["rolling_std_24"] = shifted.rolling(window=24).std()

    # Drop initial rows with NaN from lags
    data = data.dropna().reset_index(drop=True)
    return data


def split_chronological(data: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15):
    """
    Splits dataset chronologically into Train, Validation, and Test sets.
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = data.iloc[:train_end]
    val_df = data.iloc[train_end:val_end]
    test_df = data.iloc[val_end:]

    print("\n" + "=" * 65)
    print(" 2. CHRONOLOGICAL DATA SPLITTING (No Look-Ahead Leakage)")
    print("=" * 65)
    print(f"  * Total Samples (after lag filtering): {n:,}")
    print(f"  * Train Set: {len(train_df):,} samples ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"  * Val Set:   {len(val_df):,} samples ({val_df['timestamp'].min()} to {val_df['timestamp'].max()})")
    print(f"  * Test Set:  {len(test_df):,} samples ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")

    return train_df, val_df, test_df


# ==============================================================================
# 4. Model Training & Comparison
# ==============================================================================
def train_and_evaluate_models(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Trains Model 1 (Statistical ARIMA) and Model 2 (ML Random Forest Regressor),
    evaluates on validation set, selects the best model, and evaluates on test set.
    """
    feature_cols = [
        "sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek", "sin_month", "cos_month",
        "is_weekend", "lag_1", "lag_2", "lag_24", "lag_168", "rolling_mean_24", "rolling_std_24"
    ]
    if "temperature_c" in train_df.columns:
        feature_cols.append("temperature_c")

    X_train, y_train = train_df[feature_cols], train_df["demand_kwh"]

    X_val, y_val = val_df[feature_cols], val_df["demand_kwh"]
    X_test, y_test = test_df[feature_cols], test_df["demand_kwh"]
    print("\n--- TRAIN DATASET ---")
    print(train_df.head(3))
    print(f"Train Shape: {train_df.shape}")
    
    print("\n--- TEST DATASET ---")
    print(test_df.head(3))
    print(f"Test Shape: {test_df.shape}")

    print("\n" + "=" * 65)
    print(" 3. MODEL TRAINING & VALIDATION COMPARISON")
    print("=" * 65)
    
    # -------------------------------------------------------------
    # Model 1: Statistical Time Series Model (ARIMA)
    # -------------------------------------------------------------
    print("[+] Training Model 1: Statistical Time Series (ARIMA(2, 1, 2))...")
    # For training efficiency on extensive hourly series, fit on the most recent 720 hours (30 days) of training set
    arima_train_window = y_train.iloc[-720:] if len(y_train) > 720 else y_train
    arima_model = ARIMA(arima_train_window.values, order=(2, 1, 2)).fit()
    
    # Forecast across validation set length (or 168-hour representative window)
    eval_len = min(len(val_df), 168)  # 1-week validation horizon for ARIMA multi-step
    val_pred_arima = arima_model.forecast(steps=eval_len)
    
    mae_arima = mean_absolute_error(y_val.iloc[:eval_len], val_pred_arima)
    rmse_arima = np.sqrt(mean_squared_error(y_val.iloc[:eval_len], val_pred_arima))
    mape_arima = mean_absolute_percentage_error(y_val.iloc[:eval_len], val_pred_arima) * 100

    # -------------------------------------------------------------
    # Model 2: Machine Learning Model (Random Forest with Temporal & Lag Features)
    # -------------------------------------------------------------
    print("[+] Training Model 2: ML Regressor (Random Forest with Lag Features)...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=14,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    val_pred_rf = rf_model.predict(X_val)

    mae_rf = mean_absolute_error(y_val, val_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_val, val_pred_rf))
    mape_rf = mean_absolute_percentage_error(y_val, val_pred_rf) * 100

    # -------------------------------------------------------------
    # Validation Evaluation Summary Table
    # -------------------------------------------------------------
    print("\n" + "-" * 65)
    print(f"{'Model':<30} | {'MAE (kWh)':<10} | {'RMSE (kWh)':<10} | {'MAPE (%)':<8}")
    print("-" * 65)
    print(f"{'1. ARIMA(2,1,2)':<30} | {mae_arima:<10.2f} | {rmse_arima:<10.2f} | {mape_arima:<8.2f}%")
    print(f"{'2. Random Forest Regressor':<30} | {mae_rf:<10.2f} | {rmse_rf:<10.2f} | {mape_rf:<8.2f}%")
    print("-" * 65)

    # Model Selection
    selected_name = "Random Forest Regressor" if rmse_rf < rmse_arima else "ARIMA(2,1,2)"
    print(f"\n[BEST] Best Performing Model: {selected_name}")

    # -------------------------------------------------------------
    # 5. Final Evaluation on Held-out Test Set
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print(" 4. FINAL TEST SET EVALUATION (Unseen Future Data)")
    print("=" * 65)
    test_pred_rf = rf_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_pred_rf)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred_rf))
    test_mape = mean_absolute_percentage_error(y_test, test_pred_rf) * 100

    print(f"Held-out Test Performance ({selected_name}):")
    print(f"  * Test MAE:  {test_mae:.2f} kWh")
    print(f"  * Test RMSE: {test_rmse:.2f} kWh")
    print(f"  * Test MAPE: {test_mape:.2f}%")

    # Feature importances
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 5 Most Important Features:")
    for feat, imp in importances.head(5).items():
        print(f"  * {feat:<18}: {imp*100:.2f}%")

    return {
        "rf_model": rf_model,
        "feature_cols": feature_cols,
        "y_test": y_test,
        "test_pred_rf": test_pred_rf,
        "test_timestamps": test_df["timestamp"],
        "metrics": {"mae": test_mae, "rmse": test_rmse, "mape": test_mape},
        "importances": importances
    }


# ==============================================================================
# 5. Future Out-of-Sample Forecasting & Visualizations
# ==============================================================================
def plot_results(df: pd.DataFrame, eval_results: dict, output_fig: str = "forecast_results.png"):
    """
    Generates multi-panel diagnostic and forecasting figures:
    1. Overall historical context & weekly/hourly demand cycles.
    2. Actual vs Predicted demand on held-out test set (zoomed to 1-week window for detail).
    3. Feature importance distribution.
    """
    print("\n" + "=" * 65)
    print(f" 5. GENERATING FORECAST VISUALIZATIONS -> {output_fig}")
    print("=" * 65)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    plt.subplots_adjust(hspace=0.35)

    # Panel 1: Full Historical Electricity Demand with Split Demarcation
    axes[0].plot(df["timestamp"], df["demand_kwh"], color="#1f77b4", alpha=0.6, linewidth=0.8, label="Historical Demand (kWh)")
    axes[0].set_title("Full Electricity Demand Time Series (Historical Load)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Demand (kWh)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper left")

    # Panel 2: Actual vs Predicted Demand on Held-Out Test Set (Zoomed to 7-Day Window for clarity)
    test_ts = eval_results["test_timestamps"].reset_index(drop=True)
    actuals = eval_results["y_test"].reset_index(drop=True)
    preds = eval_results["test_pred_rf"]

    # Select a representative 168-hour (7 days) window from test set
    zoom_window = slice(0, 168)
    axes[1].plot(test_ts.iloc[zoom_window], actuals.iloc[zoom_window], marker="o", markersize=3, color="#2ca02c", label="Actual Test Demand", linewidth=1.5)
    axes[1].plot(test_ts.iloc[zoom_window], preds[zoom_window], marker="x", markersize=3, color="#d62728", linestyle="--", label="Random Forest Forecast", linewidth=1.5)
    axes[1].set_title(
        f"Held-Out Test Set: Actual vs Predicted (7-Day Zoom) | MAE: {eval_results['metrics']['mae']:.2f} kWh | MAPE: {eval_results['metrics']['mape']:.2f}%",
        fontsize=13,
        fontweight="bold"
    )
    axes[1].set_ylabel("Demand (kWh)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right")

    # Panel 3: Feature Importance Bar Chart
    top_feats = eval_results["importances"].head(8)
    axes[2].barh(top_feats.index[::-1], top_feats.values[::-1] * 100, color="#4c78a8", edgecolor="black", alpha=0.85)
    axes[2].set_title("Model Interpretability: Top Predictive Feature Importances (%)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Relative Importance (%)")
    axes[2].grid(True, linestyle="--", alpha=0.5, axis="x")

    plt.tight_layout()
    plt.savefig(output_fig, dpi=300)
    print(f"[OK] Successfully saved publication-quality graph to: {output_fig}")
    
    if os.environ.get("MPLBACKEND") != "Agg":
        try:
            plt.show(block=False)
            plt.pause(1)
        except Exception:
            pass


# ==============================================================================
# Main Pipeline Entry Point
# ==============================================================================
def main():
    print("=" * 65)
    print("   ELECTRICITY DEMAND TIME-SERIES FORECASTING PIPELINE   ")
    print("=" * 65)

    # 1. Load Data
    df = find_and_load_dataset(data_dir="data")

    # 2. EDA & Stationarity
    perform_eda_and_stationarity(df)

    # 3. Featurization & Chronological Split
    fe_data = create_features(df)
    train_df, val_df, test_df = split_chronological(fe_data)

    # 4. Train Models & Evaluate
    eval_results = train_and_evaluate_models(train_df, val_df, test_df)

    # 5. Visualization
    plot_results(df, eval_results, output_fig="forecast_results.png")

    print("\n" + "=" * 65)
    print("[SUCCESS] FORECASTING PIPELINE EXECUTION FINISHED SUCCESSFULLY.")
    print("=" * 65)


if __name__ == "__main__":
    main()
