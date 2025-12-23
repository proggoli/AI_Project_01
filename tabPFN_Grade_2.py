import os
import pandas as pd
import numpy as np

from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

# 1) Daten laden & Grade 2 filtern
df = pd.read_csv("data/events_with_weather.csv")
df = df[df["GRADE"] == 2].copy()


# 2) Zyklische Uhrzeit-Features
minutes = df["PLANNED_MINUTE_OF_DAY"]
df["minute_sin"] = np.sin(2 * np.pi * minutes / 1440)
df["minute_cos"] = np.cos(2 * np.pi * minutes / 1440)


# 3) Features (bei Grade 2 soll beachtet werden: S9 + Grade1 + Meta(=Wetterdaten)
features = [
    "S9_ROOT_DELAY_SEC",
    "GRADE1_DELAY_SEC",
    "minute_sin",
    "minute_cos",
    "WEEKDAY_NUM",
    "MONTH_NUM",
    "Temperatur",
    "Niederschlag",
    "Wind",
    "Schneehöhe",
]

# 4) Datum parsen + NAs entfernen
df["OPERATIONAL_DAY"] = pd.to_datetime(df["OPERATIONAL_DAY"], errors="coerce")
df = df.dropna(subset=features + ["DELAY_SEC", "OPERATIONAL_DAY"])

print("Final dataset shape:", df.shape)


# 5) Daten splitten (80/20)
df = df.sort_values("OPERATIONAL_DAY").reset_index(drop=True)
split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx]
test_df  = df.iloc[split_idx:]

X_train = train_df[features].values
X_test  = test_df[features].values

# Log-Target (negative Werte auf 0, alle verfrühte Züge erhalten 0)
y_train = np.log1p(np.clip(train_df["DELAY_SEC"].values, 0, None))
y_test  = np.log1p(np.clip(test_df["DELAY_SEC"].values, 0, None))

print("N train:", len(y_train), "N test:", len(y_test))
print("Train days:", train_df["OPERATIONAL_DAY"].min(), "→", train_df["OPERATIONAL_DAY"].max())
print("Test days :", test_df["OPERATIONAL_DAY"].min(), "→", test_df["OPERATIONAL_DAY"].max())


# 6) Modell fitten (mit GPU)
model = TabPFNRegressor(device="cuda", ignore_pretraining_limits=True)
model.fit(X_train, y_train)


# 7) Evaluation
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

mae  = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred) ** 0.5
r2   = r2_score(y_true, y_pred)

print("\n=== TabPFN Grade 2 (time-based, log-target) ===")
print("MAE :", round(mae, 2), "sec")
print("RMSE:", round(rmse, 2), "sec")
print("R²  :", round(r2, 3))



# 8) Szenario-Analysen (Grade 2)
base_row = train_df[features].median()

values = np.array([0, 60, 120, 180, 300, 600])

# Szenario A: S9 variieren, Grade 1 bleibt fix
scenario_A = pd.DataFrame([base_row] * len(values))
scenario_A["S9_ROOT_DELAY_SEC"] = values

pred_A_log = model.predict(scenario_A[features].values)
pred_A_sec = np.expm1(pred_A_log)

scenario_A_results = pd.DataFrame({
    "S9_ROOT_DELAY_SEC": values,
    "Predicted_Grade2_Delay_SEC": pred_A_sec
})

print("\n=== Szenario A: S9 → Grade 2 (Grade 1 fix) ===")
print(scenario_A_results)

# Szenario B: Grade 1 variieren, S9 fix
scenario_B = pd.DataFrame([base_row] * len(values))
scenario_B["GRADE1_DELAY_SEC"] = values

pred_B_log = model.predict(scenario_B[features].values)
pred_B_sec = np.expm1(pred_B_log)

scenario_B_results = pd.DataFrame({
    "GRADE1_DELAY_SEC": values,
    "Predicted_Grade2_Delay_SEC": pred_B_sec
})

print("\n=== Szenario B: Grade 1 → Grade 2 (S9 fix) ===")
print(scenario_B_results)


# 9) Zeitfenster-Szenario (Fixzeitpunkte) – S9 wird variiert
time_windows = {
    "HVZ_Morgen (06:30–09:00)": 7 * 60 + 30,   # 07:30
    "NVZ_Tag (09:01–16:59)":    13 * 60,       # 13:00
    "HVZ_Abend (17:00–19:00)":  18 * 60,       # 18:00
    "Nacht (19:00–06:29)":      21 * 60        # 21:00
}

rows = []
for label, minute in time_windows.items():
    minute_sin = np.sin(2 * np.pi * minute / 1440)
    minute_cos = np.cos(2 * np.pi * minute / 1440)

    scen = pd.DataFrame([base_row] * len(values))
    scen["S9_ROOT_DELAY_SEC"] = values
    scen["minute_sin"] = minute_sin
    scen["minute_cos"] = minute_cos

    pred_log = model.predict(scen[features].values)
    pred_sec = np.expm1(pred_log)

    for rd, pred in zip(values, pred_sec):
        rows.append({
            "Zeitfenster": label,
            "S9_ROOT_DELAY_SEC": rd,
            "Predicted_Grade2_Delay_SEC": pred
        })

time_window_results = pd.DataFrame(rows)

print("\n=== Szenario nach Zeitfenstern (S9 variiert) ===")
print(time_window_results)
