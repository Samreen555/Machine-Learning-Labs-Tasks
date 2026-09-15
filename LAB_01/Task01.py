import pandas as pd

# Load dataset
df = pd.read_csv("sales_data.csv")

print("========== DATASET ==========")
print(df)

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== SUMMARY STATISTICS ==========")
print(df.describe())

print("\n========== MEAN ==========")
print(df.mean(numeric_only=True))

print("\n========== MEDIAN ==========")
print(df.median(numeric_only=True))

print("New Customers:")
print(df[df["Customer_Type"] == "New"])