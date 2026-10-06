# ============================================================
# MACHINE LEARNING LAB 05
# TASK 1 - POLYNOMIAL REGRESSION
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 1. CREATE LARGE DATASET
# ============================================================

np.random.seed(42)

# 100 observations
X = np.linspace(1, 20, 100).reshape(-1, 1)

# Nonlinear relationship
y = (
    3
    + 2 * X.flatten()
    + 0.8 * (X.flatten() ** 2)
    + np.random.normal(0, 15, 100)
)

dataset = pd.DataFrame({
    "Experience": X.flatten(),
    "Salary": y
})


print("=" * 60)
print("TASK 1 - POLYNOMIAL REGRESSION")
print("=" * 60)


# ============================================================
# SCREENSHOT 1
# DATASET
# ============================================================

print("\nDataset:")
print(dataset)

print("\nFirst 10 Rows:")
print(dataset.head(10))


# ============================================================
# 2. DATASET INFORMATION
# ============================================================

print("\nDataset Shape:")
print(dataset.shape)

print("\nDataset Information:")
dataset.info()

print("\nStatistical Summary:")
print(dataset.describe())


# ============================================================
# SCREENSHOT 2
# DATASET INFORMATION
# ============================================================


# ============================================================
# 3. ORIGINAL DATA VISUALIZATION
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    X,
    y,
    color="red",
    s=40
)

plt.title("Original Dataset")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.grid(True)

plt.show()


# ============================================================
# SCREENSHOT 3
# ORIGINAL DATA GRAPH
# ============================================================


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ============================================================
# SCREENSHOT 4
# TRAIN TEST SPLIT
# ============================================================


# ============================================================
# 5. LINEAR REGRESSION - DEGREE 1
# ============================================================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_predictions = linear_model.predict(
    X_test
)

linear_mse = mean_squared_error(
    y_test,
    linear_predictions
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

print("\n" + "=" * 50)
print("DEGREE 1 - LINEAR REGRESSION")
print("=" * 50)

print("MSE:", linear_mse)
print("R2 Score:", linear_r2)


# ============================================================
# SCREENSHOT 5
# DEGREE 1 RESULTS
# ============================================================


# ============================================================
# 6. DEGREE 1 GRAPH
# ============================================================

X_grid = np.linspace(
    X.min(),
    X.max(),
    500
).reshape(-1, 1)

plt.figure(figsize=(9, 6))

plt.scatter(
    X,
    y,
    color="red",
    s=35,
    label="Actual Data"
)

plt.plot(
    X_grid,
    linear_model.predict(X_grid),
    color="blue",
    linewidth=2,
    label="Degree 1"
)

plt.title("Polynomial Regression - Degree 1")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# SCREENSHOT 6
# DEGREE 1 GRAPH
# ============================================================


# ============================================================
# 7. DEGREE 2 - QUADRATIC
# ============================================================

poly2 = PolynomialFeatures(
    degree=2
)

X_train_poly2 = poly2.fit_transform(
    X_train
)

X_test_poly2 = poly2.transform(
    X_test
)

model2 = LinearRegression()

model2.fit(
    X_train_poly2,
    y_train
)

predictions2 = model2.predict(
    X_test_poly2
)

mse2 = mean_squared_error(
    y_test,
    predictions2
)

r2_2 = r2_score(
    y_test,
    predictions2
)

print("\n" + "=" * 50)
print("DEGREE 2 - QUADRATIC REGRESSION")
print("=" * 50)

print("MSE:", mse2)
print("R2 Score:", r2_2)


# ============================================================
# SCREENSHOT 7
# DEGREE 2 RESULTS
# ============================================================


# ============================================================
# 8. DEGREE 2 GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    X,
    y,
    color="red",
    s=35,
    label="Actual Data"
)

plt.plot(
    X_grid,
    model2.predict(
        poly2.transform(X_grid)
    ),
    color="blue",
    linewidth=2,
    label="Degree 2"
)

plt.title("Polynomial Regression - Degree 2")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# SCREENSHOT 8
# DEGREE 2 GRAPH
# ============================================================


# ============================================================
# 9. DEGREE 3 - CUBIC
# ============================================================

poly3 = PolynomialFeatures(
    degree=3
)

X_train_poly3 = poly3.fit_transform(
    X_train
)

X_test_poly3 = poly3.transform(
    X_test
)

model3 = LinearRegression()

model3.fit(
    X_train_poly3,
    y_train
)

predictions3 = model3.predict(
    X_test_poly3
)

mse3 = mean_squared_error(
    y_test,
    predictions3
)

r2_3 = r2_score(
    y_test,
    predictions3
)

print("\n" + "=" * 50)
print("DEGREE 3 - CUBIC REGRESSION")
print("=" * 50)

print("MSE:", mse3)
print("R2 Score:", r2_3)


# ============================================================
# SCREENSHOT 9
# DEGREE 3 RESULTS
# ============================================================


# ============================================================
# 10. DEGREE 3 GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    X,
    y,
    color="red",
    s=35,
    label="Actual Data"
)

plt.plot(
    X_grid,
    model3.predict(
        poly3.transform(X_grid)
    ),
    color="blue",
    linewidth=2,
    label="Degree 3"
)

plt.title("Polynomial Regression - Degree 3")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# SCREENSHOT 10
# DEGREE 3 GRAPH
# ============================================================


# ============================================================
# 11. DEGREE 4 - QUARTIC
# ============================================================

poly4 = PolynomialFeatures(
    degree=4
)

X_train_poly4 = poly4.fit_transform(
    X_train
)

X_test_poly4 = poly4.transform(
    X_test
)

model4 = LinearRegression()

model4.fit(
    X_train_poly4,
    y_train
)

predictions4 = model4.predict(
    X_test_poly4
)

mse4 = mean_squared_error(
    y_test,
    predictions4
)

r2_4 = r2_score(
    y_test,
    predictions4
)

print("\n" + "=" * 50)
print("DEGREE 4 - QUARTIC REGRESSION")
print("=" * 50)

print("MSE:", mse4)
print("R2 Score:", r2_4)


# ============================================================
# SCREENSHOT 11
# DEGREE 4 RESULTS
# ============================================================


# ============================================================
# 12. DEGREE 4 GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    X,
    y,
    color="red",
    s=35,
    label="Actual Data"
)

plt.plot(
    X_grid,
    model4.predict(
        poly4.transform(X_grid)
    ),
    color="blue",
    linewidth=2,
    label="Degree 4"
)

plt.title("Polynomial Regression - Degree 4")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# SCREENSHOT 12
# DEGREE 4 GRAPH
# ============================================================


# ============================================================
# 13. DEGREE COMPARISON TABLE
# ============================================================

results = pd.DataFrame({
    "Degree": [1, 2, 3, 4],
    "MSE": [
        linear_mse,
        mse2,
        mse3,
        mse4
    ],
    "R2 Score": [
        linear_r2,
        r2_2,
        r2_3,
        r2_4
    ]
})

print("\n" + "=" * 60)
print("POLYNOMIAL DEGREE COMPARISON")
print("=" * 60)

print(results)


# ============================================================
# SCREENSHOT 13
# DEGREE COMPARISON TABLE
# ============================================================


# ============================================================
# 14. MSE COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    results["Degree"],
    results["MSE"],
    marker="o",
    linewidth=2
)

plt.title("MSE Comparison for Different Polynomial Degrees")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.xticks([1, 2, 3, 4])
plt.grid(True)

plt.show()


# ============================================================
# SCREENSHOT 14
# MSE COMPARISON GRAPH
# ============================================================


# ============================================================
# 15. R2 COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    results["Degree"],
    results["R2 Score"],
    marker="o",
    linewidth=2
)

plt.title("R² Score Comparison for Different Polynomial Degrees")
plt.xlabel("Polynomial Degree")
plt.ylabel("R² Score")
plt.xticks([1, 2, 3, 4])
plt.grid(True)

plt.show()


# ============================================================
# SCREENSHOT 15
# R2 COMPARISON GRAPH
# ============================================================


# ============================================================
# 16. FIND BEST MODEL
# ============================================================

best_index = results["MSE"].idxmin()

best_degree = int(
    results.loc[
        best_index,
        "Degree"
    ]
)

best_mse = results.loc[
    best_index,
    "MSE"
]

best_r2 = results.loc[
    best_index,
    "R2 Score"
]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Best Polynomial Degree:", best_degree)
print("Best MSE:", best_mse)
print("Best R2 Score:", best_r2)


# ============================================================
# SCREENSHOT 16
# BEST MODEL
# ============================================================


print("\n" + "=" * 60)
print("TASK 1 COMPLETED SUCCESSFULLY")
print("=" * 60)

# ============================================================
# 17. UNDERFITTING VS GOOD FIT VS OVERFITTING
# ============================================================

print("\n" + "=" * 60)
print("UNDERFITTING VS GOOD FIT VS OVERFITTING")
print("=" * 60)

# The same dataset is used so that the three polynomial
# degrees can be compared fairly.

# Degree 1 - Underfitting
model1 = LinearRegression()
model1.fit(X, y)
prediction1 = model1.predict(X_grid)

# Degree 2 - Good Fit
poly2_full = PolynomialFeatures(degree=2)
X_poly2_full = poly2_full.fit_transform(X)
X_grid_poly2_full = poly2_full.transform(X_grid)

model2_full = LinearRegression()
model2_full.fit(X_poly2_full, y)
prediction2 = model2_full.predict(X_grid_poly2_full)

# Degree 4 - Higher Complexity
poly4_full = PolynomialFeatures(degree=4)
X_poly4_full = poly4_full.fit_transform(X)
X_grid_poly4_full = poly4_full.transform(X_grid)

model4_full = LinearRegression()
model4_full.fit(X_poly4_full, y)
prediction4 = model4_full.predict(X_grid_poly4_full)

# ============================================================
# UNDERFITTING / GOOD FIT / HIGH COMPLEXITY GRAPH
# ============================================================

plt.figure(figsize=(11, 7))

plt.scatter(
    X,
    y,
    s=35,
    label="Actual Data"
)

plt.plot(
    X_grid,
    prediction1,
    linewidth=2,
    label="Degree 1 - Underfitting"
)

plt.plot(
    X_grid,
    prediction2,
    linewidth=2,
    label="Degree 2 - Good Fit"
)

plt.plot(
    X_grid,
    prediction4,
    linewidth=2,
    label="Degree 4 - High Complexity"
)

plt.title("Underfitting vs Good Fit vs High Complexity")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

# ============================================================
# SCREENSHOT 17
# UNDERFITTING VS GOOD FIT VS HIGH COMPLEXITY GRAPH
# ============================================================

print("\n" + "=" * 60)
print("TASK 1 COMPLETED SUCCESSFULLY")
print("=" * 60)
