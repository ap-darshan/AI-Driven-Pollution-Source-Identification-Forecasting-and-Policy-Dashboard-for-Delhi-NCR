import pandas as pd
from xgboost import XGBRegressor
import joblib
import numpy as np

# ✅ NEW (added)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("aqi_training_data.csv")

df["datetime"] = pd.to_datetime(df["datetime"])

# Feature engineering
df["lag1"] = df["aqi"].shift(1)
df["lag2"] = df["aqi"].shift(2)
df["lag3"] = df["aqi"].shift(3)

df["month"] = df["datetime"].dt.month
df["dayofweek"] = df["datetime"].dt.dayofweek

df = df.dropna()

features = [
    "lag1", "lag2", "lag3",
    "temperature", "humidity", "wind_speed",
    "month", "dayofweek"
]

X = df[features]
y = df["aqi"]

# ✅ NEW (added - train/test split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5
)

# ✅ FIXED (train on training data only)
model.fit(X_train, y_train)

# ✅ FIXED (predict on test data)
y_pred = model.predict(X_test)

# ✅ Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2:", round(r2, 3))

# Save model
joblib.dump(model, "xgb_model.pkl")

print("✅ Model trained and saved!")

# Save metrics
metrics = {
    "MAE": round(mae, 2),
    "RMSE": round(rmse, 2),
    "R2": round(r2, 3)
}

joblib.dump(metrics, "model_metrics.pkl")