import streamlit as st
import pandas as pd
import joblib
import datetime

st.set_page_config(page_title="Vertex", layout="wide")

# ------------------------------------------------------------
# 1. APP TITLE
# ------------------------------------------------------------
st.title("Vertex – Nifty Alpha Engine")

# ------------------------------------------------------------
# 2. LOAD PIPELINE + DATA
# ------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load("alpha_pipeline.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("test_data_streamlit.csv", parse_dates=['Date'])

pipeline = load_pipeline()
df = load_data()

st.success("✅ Model + Data Loaded")

# ------------------------------------------------------------
# 3. FEATURE LIST (MUST MATCH TRAINING)
# ------------------------------------------------------------
BASE_FEATURES = [
    'ret_5d','ret_10d','ret_21d','ret_42d','ret_63d','ret_126d',
    'rsi_14','macd_hist','bb_percentb','bb_width',
    'adx_14','atr_norm','dist_sma50','dist_sma200',
    'volume_ratio','CCI'
]

RANK_FEATURES = [
    'ret_5d_rank','ret_10d_rank','ret_21d_rank','ret_63d_rank',
    'rsi_14_rank','volume_ratio_rank','macd_hist_rank','CCI_rank'
]

FEATURES = BASE_FEATURES + RANK_FEATURES

# ------------------------------------------------------------
# 4. DATE PICKER (CALENDAR STYLE)
# ------------------------------------------------------------
st.subheader("📅 Select Date")

min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

selected_date = st.date_input(
    "Choose trading date:",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

# ------------------------------------------------------------
# 5. PREDICT BUTTON
# ------------------------------------------------------------
predict_clicked = st.button("🚀 Predict")

# ------------------------------------------------------------
# 6. RUN LOGIC ONLY AFTER BUTTON CLICK
# ------------------------------------------------------------
if predict_clicked:

    # Filter data for selected date
    day_df = df[df['Date'].dt.date == selected_date].copy()

    if day_df.empty:
        st.warning("No data for this date.")
        st.stop()

    # --------------------------------------------------------
    # 7. CREATE RANK FEATURES (MATCH TRAINING)
    # --------------------------------------------------------
    for col in [
        'ret_5d','ret_10d','ret_21d','ret_63d',
        'rsi_14','volume_ratio','macd_hist','CCI'
    ]:
        rank_col = f"{col}_rank"

        if col in day_df.columns:
            day_df[rank_col] = (
                day_df[col]
                .rank(pct=True)
                .fillna(0)
            )
        else:
            day_df[rank_col] = 0

    # --------------------------------------------------------
    # 8. ENSURE ALL FEATURES EXIST
    # --------------------------------------------------------
    for col in FEATURES:
        if col not in day_df.columns:
            day_df[col] = 0

    # --------------------------------------------------------
    # 9. RUN PREDICTIONS
    # --------------------------------------------------------
    X = day_df[FEATURES]

    try:
        probs = pipeline.predict_proba(X)[:, 1]
        day_df['prob_beat_nifty100'] = probs
        st.success("✅ Prediction completed!")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    # --------------------------------------------------------
    # 10. DISPLAY RESULTS
    # --------------------------------------------------------
    st.subheader("🚀 Top Stocks Likely to Beat Nifty")

    top_df = (
        day_df
        .sort_values("prob_beat_nifty100", ascending=False)
        .reset_index(drop=True)
    )

    st.dataframe(
        top_df[['Stock','CLOSE','prob_beat_nifty100']]
        .head(50)
        .style.format({
            'CLOSE': '₹{:.2f}',
            'prob_beat_nifty100': '{:.1%}'
        }),
        use_container_width=True
    )
