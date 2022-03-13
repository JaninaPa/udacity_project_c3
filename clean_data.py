import pandas as pd

df = pd.read_csv("census.csv")

# Info about dataframe
print(df.head(10))
print(df.describe())
print(df.isna().sum())

# Whitespace removal in column names
print(df.columns)
df.columns = df.columns.str.strip()
print(df.columns)

df.to_csv("census_cleaned.csv")