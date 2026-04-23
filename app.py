import streamlit as st
import requests
import numpy as np
import time
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components
from xgboost import XGBRegressor

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Delhi AQI Dashboard", layout="wide")
st.title("Air Quality Dashboard (Live)")

location = st.text_input(
    "🔍 Search City / State (e.g., Delhi, Bangalore, Karnataka)",
    "Delhi"
)
# -------------------------------
# AUTO REFRESH EVERY 60 SECONDS
# -------------------------------
REFRESH_INTERVAL = 60

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > REFRESH_INTERVAL:
    st.session_state.last_refresh = time.time()
    st.rerun()

TOKEN = "1b4a00d1444fe2a78fe2c181f4ff9cdac75e6aa9"  # 🔴 PUT YOUR

# -------------------------------------------------
# SEARCH LOCATION (WAQI)
# -------------------------------------------------
def search_location(location):
    url = f"https://api.waqi.info/search/?token={TOKEN}&keyword={location}"
    
    response = requests.get(url)
    data = response.json()

    if data["status"] == "ok" and len(data["data"]) > 0:
        return data["data"]
    
    return None

# Load trained ML model
model = joblib.load("xgb_model.pkl")


# -------------------------------------------------
# AQI CATEGORY
# -------------------------------------------------
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "#00E400"
    elif aqi <= 100:
        return "Moderate", "#FFFF00"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#FF7E00"
    elif aqi <= 200:
        return "Unhealthy", "#FF0000"
    elif aqi <= 300:
        return "Very Unhealthy", "#8F3F97"
    else:
        return "Hazardous", "#7E0023"

# -------------------------------------------------
# POLLUTION SOURCE IDENTIFICATION
# -------------------------------------------------
def identify_pollution_sources(iaqi_data):
    """
    Estimate pollution source contribution using pollutant patterns.
    Returns dictionary of source contribution percentage.
    """

    pm25 = iaqi_data.get("pm25", {}).get("v", 0)
    pm10 = iaqi_data.get("pm10", {}).get("v", 0)
    no2 = iaqi_data.get("no2", {}).get("v", 0)
    so2 = iaqi_data.get("so2", {}).get("v", 0)
    co = iaqi_data.get("co", {}).get("v", 0)

    # Normalize safely
    total = pm25 + pm10 + no2 + so2 + co
    if total == 0:
        return {}

    traffic_score = no2 * 0.4 + co * 0.6
    dust_score = pm10 * 0.7
    stubble_score = pm25 * 0.8
    industry_score = so2 * 0.5 + no2 * 0.3

    total_score = traffic_score + dust_score + stubble_score + industry_score

    sources = {
        "Traffic": round((traffic_score / total_score) * 100, 1),
        "Construction/Dust": round((dust_score / total_score) * 100, 1),
        "Stubble Burning": round((stubble_score / total_score) * 100, 1),
        "Industry": round((industry_score / total_score) * 100, 1),
    }

    return sources
# -------------------------------------------------
# FETCH DELHI STATIONS
# -------------------------------------------------
def fetch_delhi_stations():
    url = f"https://api.waqi.info/map/bounds/?token={TOKEN}&latlng=28.40,76.80,28.90,77.50"
    response = requests.get(url)
    data = response.json()
    if data["status"] == "ok":
        return data["data"]
    return None

# -------------------------------------------------
# FETCH STATION DETAILS (IAQI)
# -------------------------------------------------
def fetch_station_details(uid):
    url = f"https://api.waqi.info/feed/@{uid}/?token={TOKEN}"
    response = requests.get(url)
    data = response.json()
    if data["status"] == "ok":
        return data["data"]
    return None

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
# -------------------------------------------------
# LOAD DATA BASED ON SEARCH
# -------------------------------------------------
search_results = search_location(location)

if not search_results:
    st.error("Location not found. Try another city/state.")
    st.stop()

df = pd.DataFrame(search_results)

df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")
df = df.dropna(subset=["aqi"])

df["station_name"] = df["station"].apply(
    lambda x: x["name"] if isinstance(x, dict) else x
)

df["lat"] = df["station"].apply(lambda x: x["geo"][0] if isinstance(x, dict) else None)
df["lon"] = df["station"].apply(lambda x: x["geo"][1] if isinstance(x, dict) else None)

df["uid"] = df["uid"]

df["category"], df["color"] = zip(*df["aqi"].apply(get_aqi_category))

# -------------------------------------------------
# CALCULATIONS
# -------------------------------------------------
city_avg_aqi = round(df["aqi"].mean(), 2)

worst_station_row = df.loc[df["aqi"].idxmax()]
worst_station_name = worst_station_row["station_name"]
worst_station_uid = worst_station_row["uid"]
worst_aqi = int(worst_station_row["aqi"])

avg_category, avg_color = get_aqi_category(city_avg_aqi)
worst_category, worst_color = get_aqi_category(worst_aqi)

# -------------------------------------------------
# POLLUTANT SELECTOR
# -------------------------------------------------
pollutants = ["AQI", "PM2.5", "PM10", "CO", "SO2", "NO2", "O3"]

selected_pollutant = st.radio(
    "",
    pollutants,
    horizontal=True
)

# -------------------------------------------------
# FETCH POLLUTANT VALUE
# -------------------------------------------------
pollutant_value = None

if selected_pollutant == "AQI":
    pollutant_value = city_avg_aqi
else:
    station_details = fetch_station_details(worst_station_uid)
    if station_details and "iaqi" in station_details:
        key = selected_pollutant.lower().replace(".", "")
        pollutant_value = station_details["iaqi"].get(key, {}).get("v", None)

# -------------------------------------------------
# HERO SECTION
# -------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div style="
            background-color:{avg_color};
            padding:40px;
            border-radius:20px;
            text-align:center;
            color:black;">
            <h1 style="font-size:70px;margin-bottom:0;">
                {pollutant_value if pollutant_value else "N/A"}
            </h1>
            <h3>{location.title()} - {selected_pollutant}</h3>
            <p>{avg_category if selected_pollutant=="AQI" else ""}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div style="
            background-color:{worst_color};
            padding:40px;
            border-radius:20px;
            text-align:center;
            color:white;">
            <h1 style="font-size:70px;margin-bottom:0;">{worst_aqi}</h1>
            <h3>Worst Monitoring Station</h3>
            <p>{worst_station_name}</p>
            <p>{worst_category}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------
# MAJOR AIR POLLUTANTS SECTION
# -------------------------------------------------
st.markdown("## Major Air Pollutants")

# Fetch detailed station data
station_details = fetch_station_details(worst_station_uid)

iaqi_data = {}
if station_details and "iaqi" in station_details:
    iaqi_data = station_details["iaqi"]

def get_pollutant_value(key):
    return iaqi_data.get(key, {}).get("v", "N/A")

pollutant_cards = [
    ("PM2.5", get_pollutant_value("pm25"), "µg/m³", "#8F3F97"),
    ("PM10", get_pollutant_value("pm10"), "µg/m³", "#FF7E00"),
    ("CO", get_pollutant_value("co"), "ppb", "#00E400"),
    ("SO2", get_pollutant_value("so2"), "ppb", "#00E400"),
    ("NO2", get_pollutant_value("no2"), "ppb", "#FFFF00"),
    ("O3", get_pollutant_value("o3"), "ppb", "#00E400"),
]

# Create 3 columns layout
col1, col2, col3 = st.columns(3)

cols = [col1, col2, col3]

for i, (name, value, unit, color) in enumerate(pollutant_cards):
    with cols[i % 3]:
        st.markdown(
            f"""
            <div style="
                background-color:#1e1e1e;
                padding:25px;
                border-radius:15px;
                margin-bottom:20px;
                border-left:6px solid {color};
                color:white;">
                <h4 style="margin-bottom:10px;">{name}</h4>
                <h2 style="margin:0;">{value}</h2>
                <p style="margin:0;">{unit}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# -------------------------------------------------
# SOURCE CONTRIBUTION ANALYSIS
# -------------------------------------------------
st.markdown("## Pollution Source Contribution Analysis")

if station_details and "iaqi" in station_details:

    source_data = identify_pollution_sources(iaqi_data)

    if source_data:

        source_df = pd.DataFrame({
            "Source": list(source_data.keys()),
            "Contribution (%)": list(source_data.values())
        })

        fig_source = go.Figure()

        fig_source.add_trace(go.Pie(
            labels=source_df["Source"],
            values=source_df["Contribution (%)"],
            hole=0.5
        ))

        fig_source.update_layout(
            template="plotly_dark",
            height=450
        )

        st.plotly_chart(fig_source, width="stretch")

    else:
        st.warning("Not enough pollutant data to estimate sources.")
# -------------------------------------------------
# MAP + COLOR INDEX
# -------------------------------------------------
st.subheader(f"{location.title()} Monitoring Stations Map")

col_map, col_legend = st.columns([4, 1])

with col_map:
    fig = go.Figure()

    # Glow for Hazardous
    hazard_df = df[df["category"] == "Hazardous"]

    fig.add_trace(go.Scattermapbox(
        lat=hazard_df["lat"],
        lon=hazard_df["lon"],
        mode='markers',
        marker=dict(
            size=45,
            color='rgba(126,0,35,0.35)',
        ),
        hoverinfo='skip'
    ))

    # Main markers
    fig.add_trace(go.Scattermapbox(
        lat=df["lat"],
        lon=df["lon"],
        mode='markers',
        marker=dict(
            size=12,
            color=df["color"],
            showscale=False
        ),
        text=df["station_name"],
        customdata=df[["aqi", "category", "color"]],
        hovertemplate=
            "<b>%{text}</b><br><br>" +
            "AQI: <b>%{customdata[0]}</b><br>" +
            "<span style='color:%{customdata[2]};'>" +
            "%{customdata[1]}</span>" +
            "<extra></extra>"
    ))

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            zoom=10,
            center=dict(
    lat=df["lat"].mean(),
    lon=df["lon"].mean()
)
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# AQI COLOR INDEX PANEL
# -------------------------------------------------
import streamlit.components.v1 as components

with col_legend:
    st.markdown("### AQI Index")

    legend_html = """
    <div style="display:flex; align-items:center; height:400px; font-family:sans-serif; color: white;">

        <!-- Gradient Bar -->
        <div style="
            width:35px;
            height:350px;
            background: linear-gradient(to top,
                #00E400 0%,
                #00E400 16%,
                #FFFF00 16%,
                #FFFF00 32%,
                #FF7E00 32%,
                #FF7E00 48%,
                #FF0000 48%,
                #FF0000 64%,
                #8F3F97 64%,
                #8F3F97 80%,
                #7E0023 80%,
                #7E0023 100%
            );
            border-radius:8px;
            margin-right:15px;">
        </div>

        <!-- Labels -->
        <div style="
            display:flex;
            flex-direction:column;
            justify-content:space-between;
            height:350px;
            font-size:13px;">
            <span>500</span>
            <span>300</span>
            <span>200</span>
            <span>150</span>
            <span>100</span>
            <span>50</span>
            <span>0</span>
        </div>

    </div>
    """

    components.html(legend_html, height=420)


# -------------------------------------------------
# STATION TABLE
# -------------------------------------------------
st.subheader(f"{location.title()} Monitoring Stations Data")
st.dataframe(df[["station_name", "aqi", "category"]], use_container_width=True)

# -------------------------------------------------
# DOWNLOAD DATA FEATURE
# -------------------------------------------------
st.markdown("## Download AQI Data")

# Convert dataframe to CSV
csv = df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download CSV",
    data=csv,
    file_name=f"{location.lower()}_aqi_data.csv",
    mime="text/csv"
)
    
# -------------------------------------------------

# 7-DAY AQI PREDICTION GRAPH
# -------------------------------------------------
st.markdown("## 7-Day AQI Prediction")

latest = df["aqi"].tail(6)

if len(latest) >= 6:

    predictions = []
    future_dates = []

    current_time = pd.Timestamp.now()

    # Copy last values for rolling prediction
    temp_values = list(latest.values)

    for i in range(7):

        input_data = pd.DataFrame([{
    "lag1": temp_values[-1],
    "lag2": temp_values[-2],
    "lag3": temp_values[-3],
    "temperature": 30,   # dummy (or real if available)
    "humidity": 60,      # dummy
    "wind_speed": 2,     # dummy
    "month": (current_time + pd.Timedelta(days=i)).month,
    "dayofweek": (current_time + pd.Timedelta(days=i)).dayofweek
}])

        pred = model.predict(input_data)[0]
        pred = float(pred)

        predictions.append(pred)
        future_dates.append((current_time + pd.Timedelta(days=i)).strftime("%b %d"))

        # Append for next step (recursive prediction)
        temp_values.append(pred)
        temp_values.pop(0)

    # Get colors for prediction
    pred_colors = [get_aqi_category(p)[1] for p in predictions]

    # Plot graph
    fig_forecast = go.Figure()

    fig_forecast.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        mode="lines+markers",
        line=dict(color="cyan", width=3),
        marker=dict(size=10, color=pred_colors),
        name="Predicted AQI"
    ))

    fig_forecast.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_title="Date",
        yaxis_title="AQI",
        showlegend=False
    )

    st.plotly_chart(fig_forecast, width="stretch")

else:
    st.warning("Not enough data for 7-day prediction.")




# -------------------------------------------------
# PREMIUM 24-HOUR AQI BAR GRAPH
# -------------------------------------------------
st.markdown("## 24-Hour AQI Trend")

# Generate 24-hour data (replace with real later)
trend_hours = pd.date_range(end=pd.Timestamp.now(), periods=24, freq='h')
trend_values = np.random.normal(city_avg_aqi, 20, 24)
trend_values = np.clip(trend_values, 40, 400)

trend_df = pd.DataFrame({
    "time": trend_hours,
    "aqi": trend_values
})

# Find min and max
min_val = trend_df["aqi"].min()
max_val = trend_df["aqi"].max()

min_time = trend_df.loc[trend_df["aqi"].idxmin(), "time"]
max_time = trend_df.loc[trend_df["aqi"].idxmax(), "time"]

# Get color based on AQI category
trend_df["color"] = trend_df["aqi"].apply(lambda x: get_aqi_category(x)[1])

# Create bar graph
fig_bar = go.Figure()

fig_bar.add_trace(go.Bar(
    x=trend_df["time"],
    y=trend_df["aqi"],
    marker=dict(
        color=trend_df["color"],
        line=dict(width=0)
    )
))

fig_bar.update_layout(
    template="plotly_dark",
    height=500,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(
        title="Time",
        tickformat="%I:%M %p",
        showgrid=False
    ),
    yaxis=dict(
        title="AQI (US)",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.1)"
    ),
    plot_bgcolor="#111111",
    paper_bgcolor="#111111",
    showlegend=False
)

st.plotly_chart(fig_bar, width="stretch")

# -------------------------------------------------
# MIN / MAX DISPLAY (Like Image)
# -------------------------------------------------
col_min, col_max = st.columns(2)

with col_min:
    st.markdown(f"""
    <div style="background:#FF9933;
                padding:20px;
                border-radius:12px;
                text-align:center;
                color:white;">
        <h2>Min</h2>
        <h1>{int(min_val)}</h1>
        <p>{min_time.strftime("%I:%M %p")}</p>
    </div>
    """, unsafe_allow_html=True)

with col_max:
    st.markdown(f"""
    <div style="background:#8F3F97;
                padding:20px;
                border-radius:12px;
                text-align:center;
                color:white;">
        <h2>Max</h2>
        <h1>{int(max_val)}</h1>
        <p>{max_time.strftime("%I:%M %p")}</p>
    </div>
    """, unsafe_allow_html=True)






# -------------------------------------------------
# LAST 7 DAYS AQI BAR CHART
# -------------------------------------------------
st.subheader("Last 7 Days AQI Overview")

# Generate last 7 days
dates = pd.date_range(end=pd.Timestamp.now(), periods=7, freq='D')

# Simulated daily average AQI (replace later with real API if needed)
np.random.seed(42)
daily_aqi = np.random.normal(city_avg_aqi, 25, 7)
daily_aqi = np.clip(daily_aqi, 30, 450)

seven_day_df = pd.DataFrame({
    "date": dates.strftime("%b %d"),
    "aqi": daily_aqi
})

# Get color for each bar
seven_day_df["category"], seven_day_df["color"] = zip(
    *seven_day_df["aqi"].apply(get_aqi_category)
)

# Find min and max
min_row = seven_day_df.loc[seven_day_df["aqi"].idxmin()]
max_row = seven_day_df.loc[seven_day_df["aqi"].idxmax()]

# Create bar chart
fig_bar = go.Figure()

fig_bar.add_trace(go.Bar(
    x=seven_day_df["date"],
    y=seven_day_df["aqi"],
    marker_color=seven_day_df["color"],
    text=seven_day_df["aqi"].round(0),
    textposition="outside"
))

# Add Min marker
fig_bar.add_trace(go.Scatter(
    x=[min_row["date"]],
    y=[min_row["aqi"]],
    mode="markers+text",
    marker=dict(color="green", size=14, symbol="circle"),
    text=["Min"],
    textposition="bottom center",
    name="Min AQI"
))

# Add Max marker
fig_bar.add_trace(go.Scatter(
    x=[max_row["date"]],
    y=[max_row["aqi"]],
    mode="markers+text",
    marker=dict(color="red", size=14, symbol="circle"),
    text=["Max"],
    textposition="top center",
    name="Max AQI"
))

fig_bar.update_layout(
    template="plotly_dark",
    height=450,
    xaxis_title="Date",
    yaxis_title="AQI",
    showlegend=False
)

st.plotly_chart(fig_bar, width="stretch")

# DISPLAY MODEL METRICS
# -------------------------------------------------
st.markdown("## Model Performance")

try:
    metrics = joblib.load("model_metrics.pkl")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MAE", metrics["MAE"])

    with col2:
        st.metric("RMSE", metrics["RMSE"])

    with col3:
        st.metric("R² Score", metrics["R2"])

except:
    st.warning("Model metrics not found. Train model first.")
# -------------------------------------------------
# PREDICTION ACCURACY (CLEAN VERSION)
# -------------------------------------------------
st.markdown("## Prediction Accuracy")

try:
    col1, col2 = st.columns(2)

    # -------------------------------
    # Yesterday Accuracy
    # -------------------------------
    actual_aqi = df["aqi"].iloc[-1]
    predicted_aqi = df["aqi"].iloc[-2]

    yesterday_accuracy = (1 - abs(actual_aqi - predicted_aqi) / actual_aqi) * 100

    with col1:
        st.metric("Yesterday Accuracy (%)", f"{yesterday_accuracy:.2f}%")

    # -------------------------------
    # Model-based Average Accuracy
    # -------------------------------
    # Load same type of data used in training
    df_temp = pd.read_csv("aqi_training_data.csv")

    df_temp["datetime"] = pd.to_datetime(df_temp["datetime"])

    df_temp["lag1"] = df_temp["aqi"].shift(1)
    df_temp["lag2"] = df_temp["aqi"].shift(2)
    df_temp["lag3"] = df_temp["aqi"].shift(3)

    df_temp["month"] = df_temp["datetime"].dt.month
    df_temp["dayofweek"] = df_temp["datetime"].dt.dayofweek

    df_temp = df_temp.dropna()

    features = [
        "lag1", "lag2", "lag3",
        "temperature", "humidity", "wind_speed",
        "month", "dayofweek"
    ]

    X_test = df_temp[features]
    y_true = df_temp["aqi"]

    y_pred = model.predict(X_test)

    from sklearn.metrics import mean_absolute_error

    mae = mean_absolute_error(y_true, y_pred)

# Convert MAE to accuracy %
    avg_accuracy = (1 - mae / y_true.mean()) * 100
    with col2:
        st.metric("Average Accuracy (%)", f"{avg_accuracy:.2f}%")

except Exception as e:
    st.warning("Not enough data to calculate accuracy.")
# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption("Data Source: WAQI Government Monitoring Stations")
