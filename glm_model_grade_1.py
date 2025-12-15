import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv("../events_with_weather.csv")

#df.info()

# Betrachtung von Grade 1
df_g1 = df[df["GRADE"] == 1].copy()

# leere Spalten entfernen
df_g1 = df_g1.dropna(subset=["DELAY_SEC", "S9_ROOT_DELAY_SEC"])

for c in ["Temperatur", "Niederschlag", "Wind", "Schneehöhe"]:
    if c in df_g1.columns:
        df_g1[c] = df_g1[c].fillna(0)

# erster Versuch ohne Wetterdaten - um Vergleich zu haben, wie wichtig die Wetterdaten sind
glm_base = smf.glm(
    formula="""DELAY_SEC ~ S9_ROOT_DELAY_SEC + PLANNED_MINUTE_OF_DAY + WEEKDAY_NUM + MONTH_NUM""",
    data=df_g1,
    family=sm.families.Gaussian()
).fit()

print(glm_base.summary())

#Modell mit Wetterdaten
glm_weather = smf.glm(
    formula="""DELAY_SEC ~ S9_ROOT_DELAY_SEC + PLANNED_MINUTE_OF_DAY + WEEKDAY_NUM + MONTH_NUM + Temperatur + Niederschlag + Wind + Schneehöhe""",
    data=df_g1,
    family=sm.families.Gaussian()
).fit()

print(glm_weather.summary())

print("AIC Base   :", glm_base.aic)
print("AIC Weather:", glm_weather.aic)



