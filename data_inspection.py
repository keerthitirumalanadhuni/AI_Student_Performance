import pandas as pd

# Load dataset
file_path = "dataset/student_performance_10000_final.csv"

df = pd.read_csv(file_path)

print("\n===== DATASET LOADED SUCCESSFULLY =====")

print("\nDataset Shape:")
print(df.shape)

print("\n===== COLUMN NAMES =====")
for column in df.columns:
    print(column)

print("\n===== FIRST 5 RECORDS =====")
print(df.head())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE RECORDS =====")
print(df.duplicated().sum())

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())
