import os
import pandas as pd
import numpy as np

from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Daten laden - nur Daten mit Grade 1 betrachten
df = pd.read_csv("data/events_with_weather.csv")
df = df[df["GRADE"] == 1].copy()

# Zyklische Uhrzeit-Features
minutes = df["PLANNED_MINUTE_OF_DAY"]
df["minute_sin"] = np.sin(2 * np.pi * minutes / 1440)
df["minute_cos"] = np.cos(2 * np.pi * minutes / 1440)

#Features laden
features = [
    "S9_ROOT_DELAY_SEC",
    "minute_sin",
    "minute_cos",
    "WEEKDAY_NUM",
    "MONTH_NUM",
    "Temperatur",
    "Niederschlag",
    "Wind",
    "Schneehöhe",
]
target = "DELAY_SEC"

# OPERATIONAL_DAY als Datum parsen 
df["OPERATIONAL_DAY"] = pd.to_datetime(df["OPERATIONAL_DAY"], errors="coerce")

# Nur vollständige Zeilen
df = df.dropna(subset=features + [target, "OPERATIONAL_DAY"])
print("Final dataset shape:", df.shape)

# Zeitlich sortieren
df = df.sort_values("OPERATIONAL_DAY").reset_index(drop=True)

# Training auf den früheren 80% der Zeit, Test auf den 20% die späteres Datum haben
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df  = df.iloc[split_idx:]

X_train = train_df[features].values
X_test  = test_df[features].values

# Log-Target - negative Werte (also zu frühe Ankünfte werden 0 gesetzt)
y_train = np.log1p(np.clip(train_df[target].values, 0, None))
y_test  = np.log1p(np.clip(test_df[target].values, 0, None))

#Ausgabe - wie ist Modell aufgebaut
print("N train:", len(y_train), "N test:", len(y_test))
print("Train days:", train_df["OPERATIONAL_DAY"].min(), "→", train_df["OPERATIONAL_DAY"].max())
print("Test days :", test_df["OPERATIONAL_DAY"].min(), "→", test_df["OPERATIONAL_DAY"].max())

#Modell fitten, durchführen mit GPU
model = TabPFNRegressor(device="cuda", ignore_pretraining_limits=True)
model.fit(X_train, y_train)

#in Sekunden zurückrechnen
y_pred_log = model.predict(X_test)

y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

mae  = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred) ** 0.5
r2   = r2_score(y_true, y_pred)

print("\n=== TabPFN Grade 1 (time-based, log-target) ===")
print("MAE :", round(mae, 2), "sec")
print("RMSE:", round(rmse, 2), "sec")
print("R²  :", round(r2, 3))

#MAE: durchschnittliche Abweichung in Sekunden
#RMSE: empfindlicher auf Ausreisser als MAE - es gibt Fälle da liegt das Modell mehr daneben
#R2: Varianz - negativ = empfindlich, da ev Muster instabil sind, oder auch aufgrund des zeitbasierten Splits - Verspätungen haben noch andere Einflüsse

#Szenarios berechnen:

# Szenario 1: allgemein - wie viel Verspätung ergibt sich den Grade 1 Zügen bei gewisser Verspätung der S9?
# Root-Delay-Szenarien (Sekunden)
root_delays = np.array([0, 60, 120, 180, 300, 600])

# Referenz-Situation: Median aller Trainingsdaten
base_row = train_df[features].median()

scenario_df = pd.DataFrame([base_row] * len(root_delays))
scenario_df["S9_ROOT_DELAY_SEC"] = root_delays

X_scenarios = scenario_df.values

# Vorhersage (log-space)
y_scenarios_log = model.predict(X_scenarios)

# Zurück in Sekunden
y_scenarios_sec = np.expm1(y_scenarios_log)

scenario_results = pd.DataFrame({
    "S9_ROOT_DELAY_SEC": root_delays,
    "Predicted_Grade1_Delay_SEC": y_scenarios_sec
})

print("\n=== Szenario-Analyse: S9 → Grade 1 Luzern ===")
print(scenario_results)

#Szenario 2: wie sind die Unterschiede zwichen Hauptverkehrszeiten (HVZ) und Nebenverkehrszeiten (NVZ)

time_windows = {
    "HVZ_Morgen (06:30–09:00)": 7 * 60 + 30,   # 07:30
    "NVZ_Tag (09:01–16:59)":    13 * 60,       # 13:00
    "HVZ_Abend (17:00–19:00)":  18 * 60,       # 18:00
    "Nacht (19:00–06:29)":      21 * 60         # 21:00
}

root_delays = np.array([0, 60, 120, 180, 300, 600])

base_row = train_df[features].median()

scenario_rows = []

for label, minute in time_windows.items():
    minute_sin = np.sin(2 * np.pi * minute / 1440)
    minute_cos = np.cos(2 * np.pi * minute / 1440)

    scenario_df = pd.DataFrame([base_row] * len(root_delays))
    scenario_df["S9_ROOT_DELAY_SEC"] = root_delays
    scenario_df["minute_sin"] = minute_sin
    scenario_df["minute_cos"] = minute_cos

    y_log = model.predict(scenario_df[features].values)
    y_sec = np.expm1(y_log)

    for rd, pred in zip(root_delays, y_sec):
        scenario_rows.append({
            "Zeitfenster": label,
            "S9_ROOT_DELAY_SEC": rd,
            "Predicted_Grade1_Delay_SEC": pred
        })

time_window_results = pd.DataFrame(scenario_rows)

print("\n=== Szenario-Analyse nach Zeitfenstern ===")
print(time_window_results)
