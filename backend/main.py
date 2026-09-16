import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# File paths
DATA_V2_PATH = BASE_DIR / "test_data_v2.csv"
DATA_V1_PATH = BASE_DIR / "test_data_streamlit.csv"

MODEL_CLF_PATH = BASE_DIR / "model_clf.json"
MODEL_CLF_FALLBACK_PATH = BASE_DIR / "alpha_model.json"
SCALER_CLF_PATH = BASE_DIR / "scaler_clf.pkl"
SCALER_CLF_FALLBACK_PATH = BASE_DIR / "scaler.pkl"

MODEL_REG_PATH = BASE_DIR / "model_reg.json"
SCALER_REG_PATH = BASE_DIR / "scaler_reg.pkl"

PIPELINE_V1_PATH = BASE_DIR / "alpha_pipeline.pkl"

app = FastAPI(title="Vertex API", description="Nifty Alpha Engine API - Version 2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# FEATURE DEFINITIONS (Exact match to V2 Colab Notebook)
# ------------------------------------------------------------
CLF_FEATURES = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_42d', 'ret_63d', 'ret_126d',
    'rsi_14', 'macd_hist', 'bb_percentb', 'bb_width', 'adx_14', 'atr_norm',
    'dist_sma50', 'dist_sma200', 'volume_ratio', 'CCI',
    'ret_5d_rank', 'ret_10d_rank', 'ret_21d_rank', 'ret_63d_rank',
    'rsi_14_rank', 'volume_ratio_rank', 'macd_hist_rank', 'CCI_rank'
]

REG_FEATURES = [
    "ret_10d", "ret_42d", "ret_126d",
    "return_z_21", "return_z_63",
    "vol_21d", "vol_63d", "vol_126d",
    "vol_change_5d", "money_flow_volume", "volatility_of_volume",
    "CCI", "ROC_12", "ROC_26", "TSF_slope",
    "dist_52w_high", "dist_52w_low",
    "rolling_high_20d", "rolling_low_20d",
    "day_of_year", "week_of_year",
    "ret_5d", "ret_21d", "ret_63d",
    "rsi_14", "bb_percentb", "bb_width", "macd_hist", "adx_14",
    "aroon_osc", "stoch_k", "williams_r",
    "dist_sma50", "dist_sma200", "atr_norm", "volume_ratio",
    "month",
    "ret_21d_rank", "ret_63d_rank", "rsi_14_rank",
    "bb_percentb_rank", "macd_hist_rank", "volume_ratio_rank", "adx_14_rank"
]

CLIP_RULES = {
    'ret_5d': (-60, 120), 'ret_10d': (-80, 200), 'ret_21d': (-100, 300),
    'ret_42d': (-120, 400), 'ret_63d': (-150, 500), 'ret_126d': (-200, 600),
    'ROC_12': (-100, 300), 'ROC_26': (-150, 500),
    'return_z_21': (-6, 6), 'return_z_63': (-6, 6),
    'vol_21d': (0, 100), 'vol_63d': (0, 100), 'vol_126d': (0, 100),
    'volatility_of_volume': (0, None),
    'volume_ratio': (0, 15),
    'vol_change_5d': (-99, 1000),
    'money_flow_volume': (0, 1e11),
    'rsi_14': (0, 100),
    'bb_percentb': (0, 1),
    'bb_width': (0, 50),
    'CCI': (-500, 500),
    'macd_hist': (-100, 100),
    'adx_14': (0, 100),
    'atr_norm': (0, 30),
    'dist_sma50': (-60, 120), 'dist_sma200': (-80, 200),
    'dist_52w_high': (-100, 10),
    'dist_52w_low': (-10, None),
    'rolling_high_20d': (-100, 10),
    'rolling_low_20d': (-10, None),
    'month': (1, 12),
    'day_of_year': (1, 366),
    'week_of_year': (1, 53),
}

# Global loaded resources
df: Optional[pd.DataFrame] = None
model_clf: Optional[xgb.Booster] = None
scaler_clf = None
model_reg: Optional[xgb.Booster] = None
scaler_reg = None
pipeline_v1 = None

min_date_str: Optional[str] = None
max_date_str: Optional[str] = None
active_model_mode: str = "unknown"

def load_resources():
    global df, model_clf, scaler_clf, model_reg, scaler_reg, pipeline_v1
    global min_date_str, max_date_str, active_model_mode

    # 1. Load Dataset (Prefer V2 if exists, fallback to V1)
    if df is None:
        data_file = DATA_V2_PATH if DATA_V2_PATH.exists() else DATA_V1_PATH
        if not data_file.exists():
            raise FileNotFoundError(f"Neither {DATA_V2_PATH} nor {DATA_V1_PATH} found.")
        
        df = pd.read_csv(data_file, parse_dates=['Date'])
        if 'Stock' in df.columns and 'Date' in df.columns:
            df = df.sort_values(['Stock', 'Date']).reset_index(drop=True)
        
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        min_date_str = min_date.strftime("%Y-%m-%d")
        max_date_str = max_date.strftime("%Y-%m-%d")

    # 2. Load Classification Model & Scaler
    if model_clf is None:
        clf_file = MODEL_CLF_PATH if MODEL_CLF_PATH.exists() else MODEL_CLF_FALLBACK_PATH
        if clf_file.exists():
            try:
                booster = xgb.Booster()
                booster.load_model(str(clf_file))
                model_clf = booster
            except Exception as e:
                print(f"Warning: Failed to load XGBoost classifier from {clf_file}: {e}")
        
    if scaler_clf is None:
        scaler_file = SCALER_CLF_PATH if SCALER_CLF_PATH.exists() else SCALER_CLF_FALLBACK_PATH
        if scaler_file.exists():
            try:
                scaler_clf = joblib.load(scaler_file)
            except Exception as e:
                print(f"Warning: Failed to load classifier scaler: {e}")

    # 3. Load Regression Model & Scaler
    if model_reg is None and MODEL_REG_PATH.exists():
        try:
            booster = xgb.Booster()
            booster.load_model(str(MODEL_REG_PATH))
            model_reg = booster
        except Exception as e:
            print(f"Warning: Failed to load XGBoost regressor from {MODEL_REG_PATH}: {e}")

    if scaler_reg is None and SCALER_REG_PATH.exists():
        try:
            scaler_reg = joblib.load(SCALER_REG_PATH)
        except Exception as e:
            print(f"Warning: Failed to load regression scaler: {e}")

    # 4. Fallback V1 Pipeline if needed
    if (model_clf is None or scaler_clf is None) and PIPELINE_V1_PATH.exists() and pipeline_v1 is None:
        try:
            pipeline_v1 = joblib.load(PIPELINE_V1_PATH)
        except Exception as e:
            print(f"Warning: Failed to load V1 pipeline: {e}")

    if model_clf is not None and model_reg is not None:
        active_model_mode = "v2_dual_live"
    elif model_clf is not None and 'pred_return' in df.columns:
        active_model_mode = "v2_hybrid_precomputed_reg"
    elif model_clf is not None:
        active_model_mode = "v2_clf_only"
    elif pipeline_v1 is not None:
        active_model_mode = "v1_pipeline"
    else:
        active_model_mode = "data_only"

@app.on_event("startup")
def startup_event():
    load_resources()

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------
def get_nifty_features_and_predict(target_date: pd.Timestamp) -> float:
    """
    Compute approximate features for Nifty on target_date using Nifty_Close.
    Predict using existing regression model if available.
    """
    if model_reg is None or scaler_reg is None or df is None:
        return 0.0

    nifty_hist = df[df['Date'] <= target_date].groupby('Date')['Nifty_Close'].first().reset_index()
    nifty_hist = nifty_hist.sort_values('Date').reset_index(drop=True)

    if len(nifty_hist) < 20:
        return 0.0

    nifty_hist['daily_ret'] = nifty_hist['Nifty_Close'].pct_change() * 100
    for d in [5, 10, 21, 42, 63, 126]:
        nifty_hist[f'ret_{d}d'] = nifty_hist['Nifty_Close'].pct_change(d) * 100

    nifty_hist['SMA_50'] = nifty_hist['Nifty_Close'].rolling(50, min_periods=10).mean()
    nifty_hist['SMA_200'] = nifty_hist['Nifty_Close'].rolling(200, min_periods=20).mean()
    nifty_hist['dist_sma50'] = (nifty_hist['Nifty_Close'] / nifty_hist['SMA_50'] - 1) * 100
    nifty_hist['dist_sma200'] = (nifty_hist['Nifty_Close'] / nifty_hist['SMA_200'] - 1) * 100

    nifty_hist['return_z_21'] = (nifty_hist['daily_ret'] - nifty_hist['daily_ret'].mean()) / (nifty_hist['daily_ret'].std() + 1e-6)
    nifty_hist['return_z_63'] = (nifty_hist['daily_ret'] - nifty_hist['daily_ret'].rolling(63, min_periods=10).mean()) / (nifty_hist['daily_ret'].rolling(63, min_periods=10).std() + 1e-6)

    nifty_hist['vol_21d'] = nifty_hist['daily_ret'].rolling(21, min_periods=5).std()
    nifty_hist['vol_63d'] = nifty_hist['daily_ret'].rolling(63, min_periods=10).std()
    nifty_hist['vol_126d'] = nifty_hist['daily_ret'].rolling(126, min_periods=20).std()

    nifty_hist['ROC_12'] = nifty_hist['Nifty_Close'].pct_change(12) * 100
    nifty_hist['ROC_26'] = nifty_hist['Nifty_Close'].pct_change(26) * 100

    nifty_hist['TSF_slope'] = nifty_hist['Nifty_Close'].rolling(20, min_periods=5).apply(
        lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) >= 5 else 0.0
    )

    nifty_hist['month'] = nifty_hist['Date'].dt.month
    nifty_hist['day_of_year'] = nifty_hist['Date'].dt.dayofyear
    nifty_hist['week_of_year'] = nifty_hist['Date'].dt.isocalendar().week.astype(int)

    nifty_row = nifty_hist[nifty_hist['Date'] == target_date]
    if nifty_row.empty:
        nifty_row = nifty_hist.tail(1)

    nifty_features = pd.DataFrame(0.0, columns=REG_FEATURES, index=[0])

    for feat in REG_FEATURES:
        if feat in nifty_row.columns:
            val = nifty_row[feat].iloc[0]
            nifty_features[feat] = val if pd.notna(val) else 0.0

    for feat in [f for f in REG_FEATURES if '_rank' in f]:
        nifty_features[feat] = 0.5

    try:
        nifty_scaled = scaler_reg.transform(nifty_features)
        nifty_dmatrix = xgb.DMatrix(nifty_scaled, feature_names=REG_FEATURES)
        return float(model_reg.predict(nifty_dmatrix)[0])
    except Exception as e:
        print(f"Error predicting Nifty return: {e}")
        return 0.0

# ------------------------------------------------------------
# PYDANTIC RESPONSE SCHEMAS
# ------------------------------------------------------------
class PredictRequest(BaseModel):
    date: str

class StockPrediction(BaseModel):
    rank: int
    Stock: str
    CLOSE: float
    prob_beat_nifty100: float
    formatted_close: str
    formatted_prob: str

class SignalItem(BaseModel):
    rank: int
    Stock: str
    CLOSE: float
    formatted_close: str
    Beat_Nifty_Signal: str
    prob_beat_nifty100: float
    formatted_prob: str
    pred_return: Optional[float] = None
    formatted_pred_return: str
    predicted_alpha_vs_nifty: Optional[float] = None
    formatted_predicted_alpha: str

class OutcomeItem(BaseModel):
    Stock: str
    pred_return: Optional[float] = None
    formatted_pred_return: str
    actual_return_63d: Optional[float] = None
    formatted_actual_return: str
    return_deviation: Optional[float] = None
    formatted_deviation: str

class PerformanceSummary(BaseModel):
    total_stocks: int
    predicted_beat_pct: float
    beat_accuracy_pct: float
    benchmark_adjusted_accuracy_pct: float

class ShortlistItem(BaseModel):
    rank: int
    Stock: str
    CLOSE: float
    formatted_close: str
    prob_beat_nifty100: float
    formatted_prob: str
    pred_return: Optional[float] = None
    formatted_pred_return: str
    actual_return_63d: Optional[float] = None
    formatted_actual_return: str
    nifty100_ret_63d: Optional[float] = None
    formatted_nifty_return: str
    actual_alpha: Optional[float] = None
    formatted_actual_alpha: str

class PredictResponseV2(BaseModel):
    date: str
    resolved_date: str
    is_exact_date: bool
    model_version: str
    nifty_expected_return: float
    formatted_nifty_expected: str
    performance_summary: PerformanceSummary
    signals: List[SignalItem]
    outcomes: List[OutcomeItem]
    shortlist: List[ShortlistItem]
    num_shortlisted: int
    avg_actual_alpha: float
    formatted_avg_actual_alpha: str
    predictions: List[StockPrediction]

# ------------------------------------------------------------
# API ENDPOINTS
# ------------------------------------------------------------
@app.get("/api/meta")
def get_metadata():
    load_resources()
    return {
        "status": "loaded",
        "min_date": min_date_str,
        "max_date": max_date_str,
        "default_date": max_date_str,
        "total_records": len(df) if df is not None else 0,
        "unique_stocks": int(df['Stock'].nunique()) if df is not None else 0,
        "active_model_mode": active_model_mode,
        "has_v2_regression": model_reg is not None or (df is not None and 'pred_return' in df.columns)
    }

@app.post("/api/predict", response_model=PredictResponseV2)
def predict_alpha(req: PredictRequest):
    load_resources()
    try:
        user_date = datetime.strptime(req.date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Date must be in YYYY-MM-DD format")

    if df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded.")

    available_dates = df['Date'].dt.normalize()
    target_ts = pd.to_datetime(user_date)
    is_exact_date = True

    if target_ts not in available_dates.values:
        closest_ts = available_dates.iloc[(available_dates - target_ts).abs().argmin()]
        target_ts = closest_ts
        is_exact_date = False

    resolved_date_str = target_ts.strftime("%Y-%m-%d")
    day_df = df[df['Date'].dt.normalize() == target_ts].copy()

    if day_df.empty:
        raise HTTPException(status_code=404, detail="No data found for the selected trading date.")

    # --------------------------------------------------------
    # 1. Compute Cross-Sectional Ranks for Classification
    # --------------------------------------------------------
    for col in ['ret_5d','ret_10d','ret_21d','ret_63d','rsi_14','volume_ratio','macd_hist','CCI']:
        rank_col = f"{col}_rank"
        if col in day_df.columns:
            day_df[rank_col] = day_df[col].rank(pct=True).fillna(0.5)
        else:
            day_df[rank_col] = 0.5

    # --------------------------------------------------------
    # 2. Classification Prediction (prob_beat_nifty100)
    # --------------------------------------------------------
    if 'prob_beat_nifty100' not in day_df.columns or day_df['prob_beat_nifty100'].isnull().all():
        for col in CLF_FEATURES:
            if col not in day_df.columns:
                day_df[col] = 0.0

        X_clf = day_df[CLF_FEATURES].fillna(0.0)

        if model_clf is not None and scaler_clf is not None:
            try:
                X_clf_s = scaler_clf.transform(X_clf)
                dtest_clf = xgb.DMatrix(X_clf_s)
                day_df['prob_beat_nifty100'] = model_clf.predict(dtest_clf)
            except Exception as e:
                print(f"Classifier prediction error: {e}")
                day_df['prob_beat_nifty100'] = 0.5
        elif pipeline_v1 is not None:
            try:
                probs = pipeline_v1.predict_proba(X_clf)[:, 1]
                day_df['prob_beat_nifty100'] = probs
            except Exception as e:
                print(f"V1 pipeline prediction error: {e}")
                day_df['prob_beat_nifty100'] = 0.5
        else:
            day_df['prob_beat_nifty100'] = 0.5

    # --------------------------------------------------------
    # 3. Regression Prediction (pred_return 63D %)
    # --------------------------------------------------------
    nifty_pred_return = get_nifty_features_and_predict(target_ts)

    if 'pred_return' not in day_df.columns or day_df['pred_return'].isnull().all():
        if model_reg is not None and scaler_reg is not None:
            # Build regression features & clean
            for col in REG_FEATURES:
                if col not in day_df.columns:
                    day_df[col] = 0.0

            reg_data = day_df[REG_FEATURES].copy()

            # Global clipping
            for col in REG_FEATURES:
                if col.endswith('_rank'):
                    continue
                low, high = CLIP_RULES.get(col, (-500, 500))
                if low is not None:
                    reg_data[col] = reg_data[col].clip(lower=low)
                if high is not None:
                    reg_data[col] = reg_data[col].clip(upper=high)

            # Imputation
            reg_data = reg_data.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            try:
                reg_s = scaler_reg.transform(reg_data)
                dtest_reg = xgb.DMatrix(reg_s)
                day_df['pred_return'] = model_reg.predict(dtest_reg)
            except Exception as e:
                print(f"Regression prediction error: {e}")
                day_df['pred_return'] = np.nan
        else:
            if 'ret_63d' in day_df.columns:
                day_df['pred_return'] = day_df['ret_63d'].fillna(0.0)
            else:
                day_df['pred_return'] = np.nan

    # --------------------------------------------------------
    # 4. Synthesize Alpha & Signals
    # --------------------------------------------------------
    day_df['Beat_Nifty_Signal'] = day_df['prob_beat_nifty100'].apply(
        lambda x: 'Yes' if x >= 0.5 else 'No'
    )

    day_df['predicted_alpha_vs_nifty'] = day_df['pred_return'].apply(
        lambda r: (r - nifty_pred_return) if pd.notna(r) else 0.0
    )

    # Actual returns
    if 'actual_return_63d' not in day_df.columns:
        if 'stock_ret_63d' in day_df.columns:
            day_df['actual_return_63d'] = day_df['stock_ret_63d']
        elif 'target_63d_return_pct' in day_df.columns:
            day_df['actual_return_63d'] = day_df['target_63d_return_pct']
        else:
            day_df['actual_return_63d'] = np.nan

    if 'nifty100_ret_63d' not in day_df.columns:
        day_df['nifty100_ret_63d'] = np.nan

    if 'beat_nifty100' not in day_df.columns:
        valid_comp = day_df['actual_return_63d'].notna() & day_df['nifty100_ret_63d'].notna()
        day_df['beat_nifty100'] = np.nan
        day_df.loc[valid_comp, 'beat_nifty100'] = (
            day_df.loc[valid_comp, 'actual_return_63d'] > day_df.loc[valid_comp, 'nifty100_ret_63d']
        ).astype(int)

    day_df['return_deviation'] = day_df['actual_return_63d'] - day_df['pred_return']

    # Sort by probability descending
    day_df = day_df.sort_values('prob_beat_nifty100', ascending=False).reset_index(drop=True)

    # --------------------------------------------------------
    # 5. Build Performance Summary Metrics
    # --------------------------------------------------------
    total_stocks = len(day_df)
    predicted_beat_mask = (day_df['prob_beat_nifty100'] >= 0.5)
    predicted_beat_pct = float(predicted_beat_mask.mean() * 100) if total_stocks > 0 else 0.0

    valid_actual_beat = day_df['beat_nifty100'].notna()
    if valid_actual_beat.sum() > 0:
        actual_beat_mask = (day_df['beat_nifty100'] == 1)
        beat_acc = (
            (predicted_beat_mask[valid_actual_beat] == actual_beat_mask[valid_actual_beat]).mean() * 100
        )
        beat_accuracy_pct = float(beat_acc) if pd.notna(beat_acc) else 0.0

        pred_pos_count = predicted_beat_mask.sum()
        if pred_pos_count > 0:
            benchmark_correct = (
                (day_df['prob_beat_nifty100'] >= 0.5) &
                (day_df['pred_return'] > day_df['nifty100_ret_63d']) &
                (day_df['actual_return_63d'] > day_df['nifty100_ret_63d'])
            ).sum()
            benchmark_adj_acc = (benchmark_correct / pred_pos_count) * 100
            benchmark_adjusted_accuracy_pct = float(benchmark_adj_acc) if pd.notna(benchmark_adj_acc) else 0.0
        else:
            benchmark_adjusted_accuracy_pct = 0.0
    else:
        beat_accuracy_pct = 0.0
        benchmark_adjusted_accuracy_pct = 0.0

    perf_summary = PerformanceSummary(
        total_stocks=total_stocks,
        predicted_beat_pct=round(predicted_beat_pct, 2),
        beat_accuracy_pct=round(beat_accuracy_pct, 2),
        benchmark_adjusted_accuracy_pct=round(benchmark_adjusted_accuracy_pct, 2)
    )

    # --------------------------------------------------------
    # 6. Build Section 1: Signals
    # --------------------------------------------------------
    signals: List[SignalItem] = []
    predictions_v1: List[StockPrediction] = []

    for idx, row in day_df.iterrows():
        close_val = float(row.get('CLOSE', 0.0))
        prob_val = float(row.get('prob_beat_nifty100', 0.0))
        pred_ret = float(row['pred_return']) if pd.notna(row.get('pred_return')) else None
        pred_alpha = float(row['predicted_alpha_vs_nifty']) if pd.notna(row.get('predicted_alpha_vs_nifty')) else None

        sig_item = SignalItem(
            rank=idx + 1,
            Stock=str(row['Stock']),
            CLOSE=round(close_val, 2),
            formatted_close=f"₹{close_val:,.2f}",
            Beat_Nifty_Signal=str(row['Beat_Nifty_Signal']),
            prob_beat_nifty100=prob_val,
            formatted_prob=f"{prob_val:.1%}",
            pred_return=round(pred_ret, 2) if pred_ret is not None else None,
            formatted_pred_return=f"{pred_ret:+.2f}%" if pred_ret is not None else "N/A",
            predicted_alpha_vs_nifty=round(pred_alpha, 2) if pred_alpha is not None else None,
            formatted_predicted_alpha=f"{pred_alpha:+.2f}%" if pred_alpha is not None else "N/A"
        )
        signals.append(sig_item)

        if idx < 50:
            predictions_v1.append(StockPrediction(
                rank=idx + 1,
                Stock=str(row['Stock']),
                CLOSE=round(close_val, 2),
                prob_beat_nifty100=prob_val,
                formatted_close=f"₹{close_val:,.2f}",
                formatted_prob=f"{prob_val:.1%}"
            ))

    # --------------------------------------------------------
    # 7. Build Section 2: Outcomes (Prediction vs Reality)
    # --------------------------------------------------------
    outcomes: List[OutcomeItem] = []
    for idx, row in day_df.iterrows():
        pred_ret = float(row['pred_return']) if pd.notna(row.get('pred_return')) else None
        act_ret = float(row['actual_return_63d']) if pd.notna(row.get('actual_return_63d')) else None
        dev_val = float(row['return_deviation']) if pd.notna(row.get('return_deviation')) else None

        outcomes.append(OutcomeItem(
            Stock=str(row['Stock']),
            pred_return=round(pred_ret, 2) if pred_ret is not None else None,
            formatted_pred_return=f"{pred_ret:+.2f}%" if pred_ret is not None else "N/A",
            actual_return_63d=round(act_ret, 2) if act_ret is not None else None,
            formatted_actual_return=f"{act_ret:+.2f}%" if act_ret is not None else "Pending",
            return_deviation=round(dev_val, 2) if dev_val is not None else None,
            formatted_deviation=f"{dev_val:+.2f}%" if dev_val is not None else "N/A"
        ))

    # --------------------------------------------------------
    # 8. Build Section 4: Correct Predictions Shortlist
    # --------------------------------------------------------
    shortlist_mask = (
        (day_df['prob_beat_nifty100'] >= 0.5) &
        (day_df['pred_return'] > day_df['nifty100_ret_63d']) &
        (day_df['actual_return_63d'] > day_df['nifty100_ret_63d'])
    )
    shortlist_df = day_df[shortlist_mask].copy()

    shortlist: List[ShortlistItem] = []
    avg_actual_alpha = 0.0

    if not shortlist_df.empty:
        shortlist_df = shortlist_df.sort_values('actual_return_63d', ascending=False).reset_index(drop=True)
        alphas = shortlist_df['actual_return_63d'] - shortlist_df['nifty100_ret_63d']
        avg_actual_alpha = float(alphas.mean()) if not alphas.empty else 0.0

        for idx, row in shortlist_df.iterrows():
            close_val = float(row.get('CLOSE', 0.0))
            prob_val = float(row.get('prob_beat_nifty100', 0.0))
            pred_ret = float(row['pred_return']) if pd.notna(row.get('pred_return')) else None
            act_ret = float(row['actual_return_63d']) if pd.notna(row.get('actual_return_63d')) else None
            nifty_ret = float(row['nifty100_ret_63d']) if pd.notna(row.get('nifty100_ret_63d')) else None
            stock_alpha = (act_ret - nifty_ret) if (act_ret is not None and nifty_ret is not None) else None

            shortlist.append(ShortlistItem(
                rank=idx + 1,
                Stock=str(row['Stock']),
                CLOSE=round(close_val, 2),
                formatted_close=f"₹{close_val:,.2f}",
                prob_beat_nifty100=prob_val,
                formatted_prob=f"{prob_val:.1%}",
                pred_return=round(pred_ret, 2) if pred_ret is not None else None,
                formatted_pred_return=f"{pred_ret:+.2f}%" if pred_ret is not None else "N/A",
                actual_return_63d=round(act_ret, 2) if act_ret is not None else None,
                formatted_actual_return=f"{act_ret:+.2f}%" if act_ret is not None else "N/A",
                nifty100_ret_63d=round(nifty_ret, 2) if nifty_ret is not None else None,
                formatted_nifty_return=f"{nifty_ret:+.2f}%" if nifty_ret is not None else "N/A",
                actual_alpha=round(stock_alpha, 2) if stock_alpha is not None else None,
                formatted_actual_alpha=f"{stock_alpha:+.2f}%" if stock_alpha is not None else "N/A"
            ))

    return PredictResponseV2(
        date=req.date,
        resolved_date=resolved_date_str,
        is_exact_date=is_exact_date,
        model_version=active_model_mode,
        nifty_expected_return=round(nifty_pred_return, 2),
        formatted_nifty_expected=f"{nifty_pred_return:+.2f}%",
        performance_summary=perf_summary,
        signals=signals,
        outcomes=outcomes,
        shortlist=shortlist,
        num_shortlisted=len(shortlist),
        avg_actual_alpha=round(avg_actual_alpha, 2),
        formatted_avg_actual_alpha=f"{avg_actual_alpha:+.2f}%",
        predictions=predictions_v1
    )

# Mount frontend directory for static assets
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_root():
        return FileResponse(FRONTEND_DIR / "index.html")
