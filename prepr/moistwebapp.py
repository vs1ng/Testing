import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# -------------------------------------------------
# PAGE CONFIG (MUST BE FIRST STREAMLIT CALL)
# -------------------------------------------------
st.set_page_config(
    page_title="CWC Ecological Threat Indicator",
    layout="wide"
)

# -------------------------------------------------
# CORE ML LOGIC (PORTED FROM QThread)
# -------------------------------------------------
def run_prediction(csv_path, crop, date_str):
    df = pd.read_csv(csv_path)

    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month

    df['is_rain'] = (df['rain'] > 0).astype(int)
    df['moisture_level'] = (
        df['precipitation'] * 0.5
        + df['relative_humidity_2m'] * 0.2
        + 10
    )

    features = [
        'day',
        'month',
        'temperature_2m',
        'relative_humidity_2m',
        'apparent_temperature'
    ]

    X = df[features]

    m_rain = RandomForestClassifier(n_estimators=30, n_jobs=-1)
    m_moist = RandomForestRegressor(n_estimators=30, n_jobs=-1)

    m_rain.fit(X, df['is_rain'])
    m_moist.fit(X, df['moisture_level'])

    acc = accuracy_score(df['is_rain'], m_rain.predict(X))
    mae = mean_absolute_error(df['moisture_level'], m_moist.predict(X))

    day = int(date_str[:2])
    month = int(date_str[2:])
    start_date = datetime(2024, month, day)

    forecasts = []

    for i in range(14):
        pred_date = start_date + timedelta(days=i)
        pred_day = pred_date.day
        pred_month = pred_date.month

        hist = df[(df['day'] == pred_day) & (df['month'] == pred_month)]
        if hist.empty:
            hist = df[df['month'] == pred_month]

        input_row = pd.DataFrame([[
            pred_day,
            pred_month,
            hist['temperature_2m'].mean(),
            hist['relative_humidity_2m'].mean(),
            hist['apparent_temperature'].mean()
        ]], columns=features)

        rain_prob = m_rain.predict_proba(input_row)[0][1] * 100
        moist_val = m_moist.predict(input_row)[0]
        temp_val = hist['temperature_2m'].mean()

        infest_score = 0
        if 20 <= temp_val <= 35:
            infest_score += 50
        if moist_val > 14:
            infest_score += 50

        infest_prob = norm.cdf(infest_score, 50, 25) * 100

        forecasts.append({
            "day": i + 1,
            "date": pred_date,
            "rain": rain_prob,
            "moist": moist_val,
            "infest": infest_prob
        })

    return acc, mae, forecasts

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🌱 CWC Ecological Threat Indicator")

mapping_df = pd.read_csv("wdata.csv")

col1, col2, col3 = st.columns(3)

with col1:
    warehouse = st.selectbox(
        "Warehouse",
        mapping_df["WarehouseName"].tolist()
    )

with col2:
    crop = st.text_input("Crop Type")

with col3:
    date_str = st.text_input("Date (DDMM)", placeholder="e.g. 1407")

run = st.button("RUN ANALYSIS")

# -------------------------------------------------
# RUN MODEL
# -------------------------------------------------
if run:
    if not date_str.isdigit() or len(date_str) != 4:
        st.error("Date must be in DDMM format")
        st.stop()

    csv_path = mapping_df[
        mapping_df["WarehouseName"] == warehouse
    ].iloc[0]["DistrictCSVpath"]

    with st.spinner("Running model..."):
        acc, mae, forecasts = run_prediction(csv_path, crop, date_str)

    # -----------------------------
    # METRICS
    # -----------------------------
    m1, m2 = st.columns(2)
    m1.metric("Rain Model Accuracy", f"{acc:.2%}")
    m2.metric("Moisture MAE", f"{mae:.2f}")

    # -----------------------------
    # DATA TABLE
    # -----------------------------
    st.subheader("📅 14-Day Forecast")
    df_forecast = pd.DataFrame(forecasts)
    df_forecast["date"] = df_forecast["date"].dt.strftime("%b %d, %Y")
    st.dataframe(df_forecast, use_container_width=True)

    # -----------------------------
    # PLOT
    # -----------------------------
    days = df_forecast["day"]
    rain = df_forecast["rain"]
    infest = df_forecast["infest"]
    moist = df_forecast["moist"]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(days[:7], rain[:7], "o-", label="Rain", linewidth=2)
    ax.plot(days[:7], infest[:7], "o-", label="Infestation", linewidth=2)
    ax.plot(days[:7], moist[:7], "o-", label="Moisture", linewidth=2)

    ax.plot(days[7:], rain[7:], "o:", alpha=0.7)
    ax.plot(days[7:], infest[7:], "o:", alpha=0.7)
    ax.plot(days[7:], moist[7:], "o:", alpha=0.7)

    ax.axvspan(7.5, 14.5, color="red", alpha=0.15, label="Uncertainty Zone")

    ax.set_title("14-Day Threat Forecast")
    ax.set_xlabel("Days (D+N)")
    ax.set_ylabel("Probability (%) / Moisture")
    ax.set_xlim(1, 14)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend()

    st.pyplot(fig)
