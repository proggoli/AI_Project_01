import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm


events_0 = pd.read_csv(r"C:\Users\prisc\OneDrive\Dokumente\Studium\7. Semester\Artifical Intelligence\AI_Auswirkung_S9\S9_LZ.sql_2025-12-10-1116.csv", encoding="utf-8")
weather_0 = pd.read_csv(r"C:\Users\prisc\OneDrive\Dokumente\Studium\7. Semester\Artifical Intelligence\AI_Auswirkung_S9\Wetterdaten_Luzern_Jan-Dez.csv", sep=";")

events = events_0.copy()
weather = weather_0.copy()


#Zeitzonen & Datumformat der beiden Dateien anpassen
events["PLANNED_TS"] = pd.to_datetime(events["PLANNED_TS"], utc=True).dt.tz_convert("Europe/Zurich")
#Join Key auf Stundenraster erstellen für events
events["wx_time"] = events["PLANNED_TS"].dt.floor("h")

weather["reference_timestamp"] = pd.to_datetime(
    weather["reference_timestamp"],
    format="%d.%m.%Y %H:%M",
    dayfirst=True
)

weather["reference_timestamp"] = weather["reference_timestamp"].dt.tz_localize(
    "Europe/Zurich",
    nonexistent="shift_forward",
    ambiguous="NaT") #Zeitumstellung Sommerzeit beachten

#Join Key auf Stundenraster erstellen für weather
weather["wx_time"] = weather["reference_timestamp"].dt.floor("h")

#Daten von Weather auswählen, die ich brauche: htoauths=Schneehöhe Momentanwert, rre150h0=Niederschlag Stundensumme, tre005h0=Lufttemperatur Stundemittel, fu3010h = Windgeschwindigkeit Skalar Stundenmittel in km/h
weather_cols = [
    "wx_time",
    "htoauths",
    "rre150h0",
    "tre005h0",
    "fu3010h0"
]

weather_small = weather[weather_cols].copy()

#Spalten umbenenen
rename_map = {
    "htoauths": "Schneehöhe",
    "rre150h0": "Niederschlag",
    "tre005h0": "Temperatur",
    "fu3010h0": "Wind"
}

weather_small = weather_small.rename(columns={k: v for k, v in rename_map.items() if k in weather_small.columns})

#Tabellen mergen
df = events.merge(weather_small, on="wx_time", how="left")

#prüfen ob geklappt
print("Rows events:", len(events), "Rows merged:", len(df))
print("\nMissing share (Top 10)")
print(df.isna().mean().sort_values(ascending=False).head(10))
print("\nTime range events:", df["wx_time"].min(), "->", df["wx_time"].max())

#Versionen für die Modelle
df_g1 = (
    df[df["GRADE"] == 1]
    .dropna(subset=["DELAY_SEC", "S9_ROOT_DELAY_SEC"])
)

df_g2 = (
    df[df["GRADE"] == 2]
    .dropna(subset=["DELAY_SEC", "GRADE1_DELAY_SEC"])
)
#Wetterdaten bereinigen
for c in ["Schneehöhe", "Temperatur", "Niederschlag", "Wind"]:
    df_g1[c] = df_g1[c].fillna(0)
    df_g2[c] = df_g2[c].fillna(0)


