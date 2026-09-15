import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================
# CREATE SAMPLE DATA
# ==============================

data = {
    "Date": [
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
        "2026-01-04",
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
        "2026-01-10"
    ],

    "Sales": [
        5000, 3000, 2000, 5500, 3200,
        2100, 6000, 3500, 2200, 7000
    ],

    "Quantity": [
        5, 8, 6, 6, 9,
        7, 7, 10, 8, 8
    ],

    "Price": [
        1000, 375, 333, 917, 356,
        300, 857, 350, 275, 875
    ],

    "Discount": [
        5, 10, 8, 5, 12,
        10, 5, 15, 10, 5
    ]
}


# Convert dictionary into DataFrame
df = pd.DataFrame(data)


# ==============================
# DISPLAY DATA
# ==============================

print("SAMPLE DATA:")
print(df)


# ==============================
# CONVERT DATE
# ==============================

df["Date"] = pd.to_datetime(df["Date"])


# ==============================
# SELECT VARIABLES
# ==============================

data = df[
    ["Sales", "Quantity", "Price", "Discount"]
]

print("\nSELECTED VARIABLES:")
print(data)


# ==============================
# CORRELATION
# ==============================

print("\nCORRELATION:")
print(data.corr())


# ==============================
# HEATMAP
# ==============================

plt.figure(figsize=(8, 6))

sns.heatmap(
    data.corr(),
    annot=True
)

plt.title("Correlation Heatmap")

plt.show()


# ==============================
# PAIRPLOT
# ==============================

sns.pairplot(data)

plt.show()


# ==============================
# TIME-SERIES PLOT
# ==============================

plt.figure(figsize=(10, 5))

plt.plot(
    df["Date"],
    df["Sales"],
    marker="o"
)

plt.title("Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# ==============================
# ANALYSIS
# ==============================

print("\n========== ANALYSIS ==========")

print("""
1. HEATMAP:
The heatmap shows the correlation between Sales,
Quantity, Price and Discount.

2. PAIRPLOT:
The pairplot shows the relationship between every
pair of numerical variables.

3. TIME-SERIES:
The time-series graph shows how Sales change over
different dates.

4. CORRELATION:
A positive value means two variables tend to increase
together.

A negative value means one variable tends to decrease
when the other increases.

A value close to zero means there is little or no
linear relationship.
""")


print("\nTASK 02 COMPLETED!")