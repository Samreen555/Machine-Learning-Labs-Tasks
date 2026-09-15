import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# LAB TASK 02 - MATPLOTLIB
# ==========================================

# Load the Sales Dataset
df = pd.read_csv("sales_data.csv")

# Display dataset information
print("========== DATASET ==========")
print(df.head())

print("\n========== COLUMN NAMES ==========")
print(df.columns)

# Get numerical columns
numeric_columns = df.select_dtypes(include="number").columns

print("\n========== NUMERICAL COLUMNS ==========")
print(numeric_columns)


# ==========================================
# 1. LINE PLOT
# ==========================================

# Use the first numerical column for the line plot
line_column = numeric_columns[0]

plt.figure(figsize=(10, 5))

plt.plot(
    df[line_column],
    marker="o",
    linestyle="-",
    color="blue",
    label=line_column
)

plt.title("Sales Dataset - Line Plot")
plt.xlabel("Record Number")
plt.ylabel(line_column)
plt.legend()

plt.grid(True)
plt.tight_layout()
plt.show()


# ==========================================
# 2. SCATTER PLOT
# ==========================================

# Need at least two numerical columns
if len(numeric_columns) >= 2:

    x_column = numeric_columns[0]
    y_column = numeric_columns[1]

    plt.figure(figsize=(10, 5))

    plt.scatter(
        df[x_column],
        df[y_column],
        color="red",
        label=f"{x_column} vs {y_column}"
    )

    plt.title(f"{x_column} vs {y_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()

    plt.grid(True)
    plt.tight_layout()
    plt.show()

else:
    print("\nNot enough numerical columns for scatter plot.")


# ==========================================
# 3. BAR CHART
# ==========================================

# Use first 10 records for a readable bar chart
bar_data = df.head(10)

plt.figure(figsize=(10, 5))

plt.bar(
    range(len(bar_data)),
    bar_data[line_column],
    color="green",
    label=line_column
)

plt.title(f"{line_column} - Bar Chart")
plt.xlabel("Record Number")
plt.ylabel(line_column)
plt.legend()

plt.xticks(range(len(bar_data)))

plt.tight_layout()
plt.show()


# ==========================================
# ANALYSIS
# ==========================================

print("\n==========================================")
print("ANALYSIS OF VISUALIZATIONS")
print("==========================================")

print("\n1. LINE PLOT:")
print("The line plot shows how the values change from one record to another.")
print("It helps us identify increasing, decreasing, or fluctuating trends.")

print("\n2. SCATTER PLOT:")
if len(numeric_columns) >= 2:
    print(f"The scatter plot shows the relationship between {x_column} and {y_column}.")
    print("If the points move upward together, there may be a positive relationship.")
    print("If the points move downward, there may be a negative relationship.")
    print("Scattered points may indicate a weak relationship.")
else:
    print("A scatter plot could not be created because there are not enough numerical columns.")

print("\n3. BAR CHART:")
print(f"The bar chart compares the {line_column} values of the first 10 records.")
print("Higher bars represent larger values, while lower bars represent smaller values.")

print("\n==========================================")
print("TASK 02 COMPLETED")
print("==========================================")