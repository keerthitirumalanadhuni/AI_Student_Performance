import pandas as pd

# ==============================
# 1. LOAD DATASET
# ==============================

input_file = "dataset/student_performance_10000_final.csv"

df = pd.read_csv(input_file)

print("Original dataset shape:")
print(df.shape)


# ==============================
# 2. CHECK DUPLICATES
# ==============================

duplicate_count = df.duplicated().sum()

print("\nDuplicate records:", duplicate_count)

if duplicate_count > 0:
    df = df.drop_duplicates()


# ==============================
# 3. HANDLE MISSING VALUES
# ==============================

numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

print("\nMissing values BEFORE preprocessing:")
print(df[numeric_columns].isnull().sum())

for column in numeric_columns:
    if df[column].isnull().sum() > 0:
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)


print("\nMissing values AFTER preprocessing:")
print(df[numeric_columns].isnull().sum())


# ==============================
# 4. SAVE CLEAN DATASET
# ==============================

output_file = "dataset/student_performance_clean.csv"

df.to_csv(output_file, index=False)

print("\nClean dataset saved successfully.")
print("Output file:", output_file)

print("\nFinal dataset shape:")
print(df.shape)
