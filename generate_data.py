import pandas as pd
import numpy as np

# Generate hourly timestamps
dates = pd.date_range(start="2024-01-01", periods=1000, freq="h")

data = []

for dt in dates:
    # Simulate AQI pattern (day-night cycle)
    base_aqi = 150 + 60 * np.sin(dt.hour / 24 * 2 * np.pi)

    aqi = base_aqi + np.random.randint(-25, 25)

    # Simulated weather
    temp = 25 + 10 * np.sin(dt.hour / 24 * 2 * np.pi)
    humidity = 50 + np.random.randint(-15, 20)
    wind = np.random.uniform(1, 6)

    data.append([dt, round(aqi, 2), round(temp, 2), humidity, round(wind, 2)])

df = pd.DataFrame(data, columns=[
    "datetime", "aqi", "temperature", "humidity", "wind_speed"
])

df.to_csv("aqi_training_data.csv", index=False)

print("✅ Dataset generated successfully!")