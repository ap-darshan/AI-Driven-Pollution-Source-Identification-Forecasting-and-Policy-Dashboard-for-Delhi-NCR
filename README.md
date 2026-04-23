# 🌍 AI-Based Air Quality Prediction Dashboard

## 📌 Overview
This project predicts Air Quality Index (AQI) using Machine Learning and displays real-time pollution data using an interactive dashboard.

## 🚀 Features
- Live AQI data using WAQI API
- Pollution source analysis
- 7-day AQI prediction using XGBoost
- Interactive dashboard using Streamlit
- Data visualization using Plotly

## 🧠 Tech Stack
- Python
- Streamlit
- XGBoost
- Pandas, NumPy
- Plotly

## 📊 Machine Learning
- Model: XGBoost Regressor
- Features:
  - Lag values (lag1, lag2, lag3)
  - Temperature
  - Humidity
  - Wind speed
  - Month & Day of Week

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

## 📸 Project Screenshots

### Dashboard Overview
![Dashboard](screenshots/<img width="1880" height="595" alt="dashboard" src="https://github.com/user-attachments/assets/a5628459-4caa-428f-bb73-d30595b00af8" />
)

### AQI Monitoring Map
![Map](screenshots/map.png)

### Pollution Source Analysis
![Sources](screenshots/sources.png)

### 7-Day AQI Prediction
![Prediction](screenshots/prediction.png)

### 24-Hour AQI Trend
![Trend](screenshots/trend.png)
