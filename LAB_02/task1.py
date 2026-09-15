import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =================================
# CREATE SAMPLE DATA
# =================================

data = {
    "Product": ["Laptop", "Phone", "Tablet", "Laptop", "Phone",
                "Tablet", "Laptop", "Phone", "Tablet", "Laptop"],

    "Sales": [5000, 3000, 2000, 5500, 3200,
              2100, 6000, 3500, 2200, 7000],

    "Quantity": [5, 8, 6, 6, 9,
                 7, 7, 10, 8, 8]
}

df = pd.DataFrame(data)


# =================================
# DISPLAY DATA
# =================================

print("SAMPLE DATA:")
print(df)

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nSHAPE:")
print(df.shape)

print("\nDATA TYPES:")
print(df.dtypes)


# =================================
# SUMMARY STATISTICS
# =================================

print("\nSUMMARY STATISTICS:")
print(df.describe())

print("\nMEAN:")
print(df.mean(numeric_only=True))

print("\nMEDIAN:")
print(df.median(numeric_only=True))

print("\nSTANDARD DEVIATION:")
print(df.std(numeric_only=True))


# =================================
# CHECK MISSING VALUES
# =================================

print("\nMISSING VALUES:")
print(df.isnull().sum())


# =================================
# CHECK DUPLICATES
# =================================

print("\nDUPLICATES:")
print(df.duplicated().sum())


# =================================
# HISTOGRAM
# =================================

plt.hist(df["Sales"], bins=5)

plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")

plt.show()


# =================================
# ADD AN OUTLIER
# =================================

df_outlier = df.copy()

df_outlier.loc[0, "Sales"] = 50000



# =================================
# BOX PLOT
# =================================

sns.boxplot(y=df_outlier["Sales"])

plt.title("Sales - Outlier Detection")
plt.ylabel("Sales")

plt.show()


# =================================
# IQR OUTLIER DETECTION
# =================================

Q1 = df_outlier["Sales"].quantile(0.25)
Q3 = df_outlier["Sales"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df_outlier[
    (df_outlier["Sales"] < lower) |
    (df_outlier["Sales"] > upper)
]

print("\nOUTLIERS:")
print(outliers)

print("\nNumber of Outliers:", len(outliers))


# =================================
# SCATTER PLOT
# =================================

plt.scatter(
    df["Quantity"],
    df["Sales"]
)

plt.title("Quantity vs Sales")
plt.xlabel("Quantity")
plt.ylabel("Sales")

plt.show()


print("\nTASK 01 COMPLETED!")