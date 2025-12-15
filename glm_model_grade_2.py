import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Daten laden (eine Ebene höher)
df = pd.read_csv("../events_with_weather.csv")


# Datenaufbereitung nur für Grade 2
df_g2 = df[df["GRADE"] == 2].copy()

df_g2 = df_g2.dropna(subset=["DELAY_SEC", "GRADE1_DELAY_SEC"])

for c in ["Temperatur", "Niederschlag", "Wind", "Schneehöhe"]:
    if c in df_g2.columns:
        df_g2[c] = df_g2[c].fillna(0)

glm2_base = smf.glm(
    formula="""DELAY_SEC ~ GRADE1_DELAY_SEC + PLANNED_MINUTE_OF_DAY + WEEKDAY_NUM + MONTH_NUM """,
    data=df_g2,
    family=sm.families.Gaussian()
).fit()

print(glm2_base.summary())

glm2_weather = smf.glm(
    formula="""DELAY_SEC ~ GRADE1_DELAY_SEC + PLANNED_MINUTE_OF_DAY + WEEKDAY_NUM + MONTH_NUM + Temperatur + Niederschlag + Wind + Schneehöhe""",
    data=df_g2,
    family=sm.families.Gaussian()
).fit()

print(glm2_weather.summary())

print("AIC Grade 2 Base   :", glm2_base.aic)
print("AIC Grade 2 Weather:", glm2_weather.aic)
