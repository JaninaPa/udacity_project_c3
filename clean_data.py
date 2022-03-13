import pandas as pd

df = pd.read_csv("census.csv")

# Info about dataframe
print(df.head(10))
print(df.shape)
print(df.describe())
print(df.isna().sum())

# Whitespace removal in column names
print(df.columns)
df.columns = df.columns.str.strip()
print(df.columns)

# The slice testing showed, that some entries contain question marks.
# Removal of '?'
list_cat_cols = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
for col in list_cat_cols:
    df = df.drop(df[df[col].str.contains("?", regex=False)].index)
print(df.shape)

df.to_csv("census_cleaned.csv")