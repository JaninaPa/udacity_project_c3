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

for col in df.columns:
    print(df[col].unique())

# The results of slice testing and display of column names showed, 
# that some entries contain question marks and leading whitespaces.

# Removal of '?' and leading whitespaces
list_cat_cols = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "salary"
]
for col in list_cat_cols:
    df = df.drop(df[df[col].str.contains("?", regex=False)].index)
    df[col] = df[col].str.strip()
print(df.shape)
for col in df.columns:
    print(df[col].unique())

print(len(df[df.salary == '<=50K']))
print(len(df[df.salary == '>50K']))

df.to_csv("census_cleaned.csv")