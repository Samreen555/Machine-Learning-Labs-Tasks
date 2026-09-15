import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("sales_data.csv")


# =================================
# SELECT IMPORTANT VARIABLES
# =================================

data = df[
    [
        "Sales_Amount",
        "Quantity_Sold",
        "Unit_Cost",
        "Unit_Price",
        "Discount"
    ]
]

print("SELECTED DATA:")
print(data.head())


# =================================
# 1. CORRELATION
# =================================

print("\nCORRELATION:")
print(data.corr())


# =================================
# 2. HEATMAP
# =================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    data.corr(),
    annot=True
)

plt.title("Correlation Heatmap")

plt.show()


# =================================
# 3. PAIRPLOT
# =================================

sns.pairplot(data)

plt.show()


# =================================
# 4. TIME SERIES
# =================================

df["Sale_Date"] = pd.to_datetime(df["Sale_Date"])

df = df.sort_values("Sale_Date")


plt.figure(figsize=(10, 5))

plt.plot(
    df["Sale_Date"],
    df["Sales_Amount"]
)

plt.title("Sales Amount Over Time")
plt.xlabel("Sale Date")
plt.ylabel("Sales Amount")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()
