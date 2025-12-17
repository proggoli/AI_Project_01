import pandas as pd
import numpy as np

from tabpfn import TabPFNRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 1) Daten laden
df = pd.read_csv("../events_with_weather.csv")

# 2) Grade 1 filtern
df_g1 = df[df["GRADE"] == 1].copy()

features = [
    "S9_ROOT_DELAY_SEC",
    "PLANNED_MINUTE_OF_DAY",
    "WEEKDAY_NUM",
    "MONTH_NUM",
    "Temperatur",
    "Niederschlag",
    "Wind",
    "Schneehöhe",
] 

target = "DELAY_SEC"

# Nur Zeilen mit allen nötigen Werten
df_g1 = df_g1.dropna(subset=features + [target])

X = df_g1[features].values
y = df_g1[target].values

# 3) Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4) Scaling (hilft oft)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5) Modell trainieren
model = TabPFNRegressor(device="cpu")  # falls du GPU hast: "cuda"
model.fit(X_train, y_train)

# 6) Predict + Metrics
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("=== TabPFN Grade 1 ===")
print("N train:", len(y_train), "N test:", len(y_test))
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 3))

