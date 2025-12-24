import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import numpy as np

# -------------------------------------------------
# PAGE CONFIG (MUST BE FIRST STREAMLIT CALL)
# -------------------------------------------------
st.set_page_config(
    page_title="CWC Ecological Threat Indicator",
    layout="wide"
)

# Custom CSS for colored warehouse boxes
st.markdown("""
<style>
.blue-warehouse {
    background-color: rgba(33, 150, 243, 0.2);
    border-left: 4px solid #2196F3;
    padding: 8px;
    margin: 4px 0;
    border-radius: 4px;
}
.red-warehouse {
    background-color: rgba(244, 67, 54, 0.2);
    border-left: 4px solid #F44336;
    padding: 8px;
    margin: 4px 0;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# DATA LOADING
# -------------------------------------------------
@st.cache_data
def load_warehouse_data():
    """Load both warehouse datasets"""
    mapping_df = pd.read_csv("wdata.csv")
    geocode_df = pd.read_csv("geocode.csv")
    
    csv_warehouses = set(mapping_df["WarehouseName"].tolist())
    geocode_warehouses = set(geocode_df["WarehouseName"].tolist())
    
    # Create combined list with type indicator
    all_warehouses = []
    for wh in sorted(csv_warehouses):
        all_warehouses.append({"name": wh, "type": "csv"})
    for wh in sorted(geocode_warehouses):
        all_warehouses.append({"name": wh, "type": "geocode"})
    
    return mapping_df, geocode_df, all_warehouses

# -------------------------------------------------
# OPEN-METEO API FUNCTIONS
# -------------------------------------------------
def fetch_openmeteo_data(lat, lon, start_date, days=14):
    """
    Fetch historical and forecast weather data from Open-Meteo API
    
    Args:
        lat: Latitude
        lon: Longitude
        start_date: Starting date for forecast
        days: Number of days to fetch
    
    Returns:
        DataFrame with weather data or None if error
    """
    debug_info = {
        "url": "",
        "params": {},
        "status_code": None,
        "error": None,
        "response": None,
        "endpoint_used": ""
    }
    
    try:
        # Calculate date range
        end_date = start_date + timedelta(days=days-1)
        today = datetime.now().date()
        
        # Determine which endpoint to use
        # If start_date is in the past, use historical-weather endpoint
        # If start_date is today or future, use forecast endpoint
        if start_date.date() < today:
            url = "https://api.open-meteo.com/v1/historical-weather"
            debug_info["endpoint_used"] = "historical-weather (past dates)"
        else:
            url = "https://api.open-meteo.com/v1/forecast"
            debug_info["endpoint_used"] = "forecast (future dates)"
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum",
            "timezone": "auto"
        }
        
        debug_info["url"] = url
        debug_info["params"] = params
        
        response = requests.get(url, params=params, timeout=10)
        debug_info["status_code"] = response.status_code
        
        response.raise_for_status()
        data = response.json()
        debug_info["response"] = data
        
        # Parse response
        daily = data.get("daily", {})
        
        df = pd.DataFrame({
            "date": pd.to_datetime(daily["time"]),
            "temperature_2m": daily["temperature_2m_mean"],
            "relative_humidity_2m": daily["relative_humidity_2m_mean"],
            "precipitation": daily["precipitation_sum"]
        })
        
        # Add calculated fields
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['apparent_temperature'] = df['temperature_2m']  # Simplified
        df['rain'] = df['precipitation']
        df['is_rain'] = (df['rain'] > 0).astype(int)
        df['moisture_level'] = (df['precipitation'] * 0.5 + df['relative_humidity_2m'] * 0.2 + 10)
        
        return df, debug_info
        
    except requests.exceptions.RequestException as e:
        debug_info["error"] = f"Request Error: {str(e)}"
        return None, debug_info
    except KeyError as e:
        debug_info["error"] = f"Data Parsing Error: {str(e)}"
        return None, debug_info
    except Exception as e:
        debug_info["error"] = f"Unexpected Error: {str(e)}"
        return None, debug_info

def analyze_api_forecast(df):
    """
    Analyze weather data from API to generate threat forecasts
    
    Args:
        df: DataFrame with weather data
    
    Returns:
        List of forecast dictionaries
    """
    forecasts = []
    
    for i, row in df.iterrows():
        rain_prob = row['is_rain'] * 100  # Simple: 100% if rain, 0% if not
        moist_val = row['moisture_level']
        temp_val = row['temperature_2m']
        
        # Infestation logic (same as original)
        infest_score = 0
        if 20 <= temp_val <= 35:
            infest_score += 50
        if moist_val > 14:
            infest_score += 50
        
        infest_prob = norm.cdf(infest_score, 50, 25) * 100
        
        forecasts.append({
            "day": i + 1,
            "date": row['date'],
            "rain": rain_prob,
            "moist": moist_val,
            "infest": infest_prob,
            "temp": temp_val,
            "humidity": row['relative_humidity_2m'],
            "precipitation": row['precipitation']
        })
    
    return forecasts

# -------------------------------------------------
# CORE ML LOGIC (FOR CSV WAREHOUSES)
# -------------------------------------------------
def run_prediction(csv_path, crop, date_str):
    """Original ML prediction for CSV warehouses - year-agnostic pattern matching"""
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

    forecasts = []

    for i in range(14):
        # Calculate the predicted day/month (wrapping around month boundaries)
        pred_day = day + i
        pred_month = month
        
        # Handle month overflow
        while pred_day > 31:  # Simplified - good enough for most cases
            pred_day -= 30
            pred_month += 1
            if pred_month > 12:
                pred_month = 1
        
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
            "date_day": pred_day,
            "date_month": pred_month,
            "rain": rain_prob,
            "moist": moist_val,
            "infest": infest_prob
        })

    return acc, mae, forecasts

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🌱 CWC Ecological Threat Indicator")

# Load data
mapping_df, geocode_df, all_warehouses = load_warehouse_data()

# Display warehouse legend
col_leg1, col_leg2 = st.columns(2)
with col_leg1:
    st.markdown("🔵 **Blue**: CSV Historical Data (Pattern-based ML)")
with col_leg2:
    st.markdown("🔴 **Red**: Live API Data (Real-time Forecast)")

st.markdown("---")

# Create warehouse selector with colored boxes
st.subheader("Select Warehouse")

# Create a custom selectbox with colored display
warehouse_options = [wh["name"] for wh in all_warehouses]
warehouse_types = {wh["name"]: wh["type"] for wh in all_warehouses}

selected_warehouse = st.selectbox(
    "Warehouse",
    warehouse_options,
    format_func=lambda x: f"{'🔵' if warehouse_types[x] == 'csv' else '🔴'} {x}"
)

warehouse_type = warehouse_types[selected_warehouse]

# Show warehouse type indicator
if warehouse_type == "csv":
    st.info(f"🔵 **{selected_warehouse}** - Using historical CSV data (15 years of patterns)")
else:
    st.warning(f"🔴 **{selected_warehouse}** - Using Open-Meteo API (live forecast)")

# Input fields
col1, col2 = st.columns(2)

with col1:
    crop = st.text_input("Crop Type", placeholder="e.g., Wheat, Rice")

with col2:
    if warehouse_type == "csv":
        date_str = st.text_input("Start Date (DDMM)", placeholder="e.g. 1407")
    else:
        # For API warehouses, disable date input and show auto message
        date_str = st.text_input(
            "Start Date (DDMM)", 
            value="AUTO",
            disabled=True,
            help="API warehouses use live data starting from today"
        )
        st.caption("📅 Forecast will run for the next 14 days from today")

run = st.button("🚀 RUN ANALYSIS", type="primary")

# -------------------------------------------------
# RUN MODEL
# -------------------------------------------------
if run:
    # Validation for CSV warehouses only
    if warehouse_type == "csv":
        if not date_str.isdigit() or len(date_str) != 4:
            st.error("❌ Date must be in DDMM format (e.g., 1407 for July 14)")
            st.stop()

        day = int(date_str[:2])
        month = int(date_str[2:])
        
        # Validate day and month
        if not (1 <= day <= 31 and 1 <= month <= 12):
            st.error("❌ Invalid date. Day must be 1-31, Month must be 1-12.")
            st.stop()

    debug_info = None
    
    # -------------------------------------------------
    # BRANCH: CSV WAREHOUSE
    # -------------------------------------------------
    if warehouse_type == "csv":
        matched = mapping_df[mapping_df["WarehouseName"] == selected_warehouse]
        if matched.empty:
            st.error(f"❌ Warehouse '{selected_warehouse}' not found in wdata.csv")
            st.stop()
        csv_path = matched.iloc[0]["DistrictCSVpath"]
        if pd.isna(csv_path):
            st.error(f"❌ No CSV path configured for warehouse '{selected_warehouse}'")
            st.stop()
        import os
        if not os.path.exists(csv_path):
            st.error(f"❌ CSV file not found: {csv_path}")
            st.stop()
        with st.spinner("Running ML model on historical patterns (2010-2024)..."):
            acc, mae, forecasts = run_prediction(csv_path, crop, date_str)
    # -------------------------------------------------
    # BRANCH: GEOCODE WAREHOUSE (API)
    # -------------------------------------------------
    else:
        warehouse_info = geocode_df[geocode_df["WarehouseName"] == selected_warehouse].iloc[0]
        lat = warehouse_info["lat"]
        lon = warehouse_info["long"]
        district = warehouse_info["District"]
        state = warehouse_info["State"]
        
        st.info(f"📍 Location: {district}, {state} (Lat: {lat:.4f}, Lon: {lon:.4f})")
        
        # Use current date for API query
        start_date = datetime.now()
        
        st.info(f"🗓️ Fetching forecast from {start_date.strftime('%B %d, %Y')} for the next 14 days")
        
        with st.spinner("Fetching live weather data from Open-Meteo API..."):
            weather_df, debug_info = fetch_openmeteo_data(lat, lon, start_date, days=14)
        
        if weather_df is None:
            st.error("❌ Failed to fetch weather data from API. Check debug info below.")
            forecasts = None
        else:
            st.success("✅ Weather data fetched successfully!")
            forecasts = analyze_api_forecast(weather_df)
            
            # Show simple stats instead of ML metrics
            st.subheader("📊 Weather Data Summary")
            m1, m2, m3 = st.columns(3)
            m1.metric("Avg Temperature", f"{weather_df['temperature_2m'].mean():.1f}°C")
            m2.metric("Avg Humidity", f"{weather_df['relative_humidity_2m'].mean():.1f}%")
            m3.metric("Total Precipitation", f"{weather_df['precipitation'].sum():.1f}mm")

    # -------------------------------------------------
    # DISPLAY RESULTS (COMMON FOR BOTH)
    # -------------------------------------------------
    if forecasts:
        st.markdown("---")
        
        # DATA TABLE
        st.subheader("📅 14-Day Forecast")
        df_forecast = pd.DataFrame(forecasts)
        
        # Format display columns differently based on warehouse type
        if warehouse_type == "csv":
            # For CSV: Show only day/month (no year)
            month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                          7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
            df_forecast["date_display"] = df_forecast.apply(
                lambda row: f"{month_names[row['date_month']]} {row['date_day']}", axis=1
            )
            display_cols = ["day", "date_display", "rain", "infest", "moist"]
            df_display = df_forecast[display_cols].copy()
            df_display.columns = ["Day", "Date (Pattern)", "Rain Prob (%)", "Infestation (%)", "Moisture Level"]
        else:
            # For API: Show full date with year
            df_forecast["date_display"] = pd.to_datetime(df_forecast["date"]).dt.strftime("%b %d, %Y")
            display_cols = ["day", "date_display", "rain", "infest", "moist"]
            df_display = df_forecast[display_cols].copy()
            df_display.columns = ["Day", "Date (Forecast)", "Rain Prob (%)", "Infestation (%)", "Moisture Level"]
        
        st.dataframe(df_display, use_container_width=True)

        # PLOT
        st.subheader("📈 Threat Visualization")
        
        days = df_forecast["day"]
        rain = df_forecast["rain"]
        infest = df_forecast["infest"]
        moist = df_forecast["moist"]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Solid lines for first 7 days
        ax.plot(days[:7], rain[:7], "o-", label="Rain Probability", 
                linewidth=2.5, markersize=8, color='#2196F3')
        ax.plot(days[:7], infest[:7], "o-", label="Infestation Risk", 
                linewidth=2.5, markersize=8, color='#FF9800')
        ax.plot(days[:7], moist[:7], "o-", label="Moisture Level", 
                linewidth=2.5, markersize=8, color='#4CAF50')

        # Dotted lines for days 8-14
        ax.plot(days[7:], rain[7:], "o:", alpha=0.7, linewidth=2.5, 
                markersize=8, color='#2196F3')
        ax.plot(days[7:], infest[7:], "o:", alpha=0.7, linewidth=2.5, 
                markersize=8, color='#FF9800')
        ax.plot(days[7:], moist[7:], "o:", alpha=0.7, linewidth=2.5, 
                markersize=8, color='#4CAF50')

        # Uncertainty zone
        ax.axvspan(7.5, 14.5, color="red", alpha=0.15, label="Uncertainty Zone")

        title_suffix = "(Historical Patterns)" if warehouse_type == "csv" else "(Live Forecast)"
        ax.set_title(f"14-Day Threat Forecast - {selected_warehouse} {title_suffix}", 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel("Days (D+N)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Probability (%) / Moisture Level", fontsize=12, fontweight='bold')
        ax.set_xlim(0.5, 14.5)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=10)

        st.pyplot(fig)
        
        # Download option
        download_filename = f"forecast_{selected_warehouse}_{datetime.now().strftime('%Y%m%d')}.csv" if warehouse_type == "geocode" else f"forecast_{selected_warehouse}_{date_str}.csv"
        st.download_button(
            "📥 Download Forecast Data (CSV)",
            df_display.to_csv(index=False).encode('utf-8'),
            download_filename,
            "text/csv"
        )

    # -------------------------------------------------
    # DEBUG WINDOW (FOR API WAREHOUSES)
    # -------------------------------------------------
    if warehouse_type == "geocode" and debug_info:
        st.markdown("---")
        with st.expander("🔍 API Debug Information", expanded=(forecasts is None)):
            st.subheader("Request Details")
            st.json({
                "Endpoint Used": debug_info["endpoint_used"],
                "API URL": debug_info["url"],
                "Parameters": debug_info["params"],
                "Status Code": debug_info["status_code"]
            })
            
            if debug_info["error"]:
                st.error(f"**Error**: {debug_info['error']}")
            
            if debug_info["response"]:
                st.subheader("API Response")
                st.json(debug_info["response"])
            
            st.info("💡 **Tip**: If you see errors, check your internet connection or verify the coordinates are correct.")
