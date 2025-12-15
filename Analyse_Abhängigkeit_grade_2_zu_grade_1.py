import pandas as pd

df = pd.read_csv("../events_with_weather.csv")

print("Total rows:", len(df))
print("\nCounts by grade:")
print(df["GRADE"].value_counts().sort_index())

