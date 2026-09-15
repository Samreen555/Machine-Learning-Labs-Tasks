import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("sales_data.csv")


# ==============================
# DATASET INFORMATION
# ==============================

print("FIRST 5 ROWS:")
print(df.head())

print("\nDATASET SHAPE:")
print(df.shape)

print("\nDATA TYPES:")
print(df.dtypes)


# ==============================
# SUMMARY STATISTICS
# ==============================

print("\nSUMMARY STATISTICS:")
print(df.describe())

print("\nMEAN:")
print(df.mean(numeric_only=True))

print("\nMEDIAN:")
print(df.median(numeric_only=True))

print("\nSTANDARD DEVIATION:")
print(df.std(numeric_only=True))


# # ==============================
# # CUSTOMER TYPE
# # ==============================

# print("\nNEW CUSTOMERS:")
# print((df["Customer_Type"] == "New").sum())

# print("\nRETURNING CUSTOMERS:")
# print((df["Customer_Type"] == "Returning").sum())


# ==============================
# HISTOGRAM
# ==============================

plt.figure(figsize=(8, 5))

plt.hist(df["Sales_Amount"], bins=20)

plt.title("Sales Amount Distribution")
plt.xlabel("Sales Amount")
plt.ylabel("Frequency")

plt.show()


# ==============================
# ADD ONE OUTLIER
# ==============================

# Create a copy so original data is safe
df_eda = df.copy()

# Add one extreme value
df_eda.loc[0, "Sales_Amount"] = 50000


# ==============================
# BOX PLOT
# ==============================

plt.figure(figsize=(8, 5))

sns.boxplot(y=df_eda["Sales_Amount"])

plt.title("Sales Amount - Outlier Detection")
plt.ylabel("Sales Amount")

plt.show()


# ==============================
# IQR OUTLIER DETECTION
# ==============================

Q1 = df_eda["Sales_Amount"].quantile(0.25)
Q3 = df_eda["Sales_Amount"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df_eda[
    (df_eda["Sales_Amount"] < lower_limit) |
    (df_eda["Sales_Amount"] > upper_limit)
]

print("\nOUTLIERS:")
print(outliers[["Product_ID", "Sales_Amount"]])


# ==============================
# SCATTER PLOT
# ==============================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Quantity_Sold"],
    df["Sales_Amount"]
)

plt.title("Quantity Sold vs Sales Amount")
plt.xlabel("Quantity Sold")
plt.ylabel("Sales Amount")

plt.show()


# ==============================
# CUSTOMER TYPE BAR CHART
# ==============================

df["Customer_Type"].value_counts().plot(kind="bar")

plt.title("New vs Returning Customers")
plt.xlabel("Customer Type")
plt.ylabel("Number of Customers")

plt.show()