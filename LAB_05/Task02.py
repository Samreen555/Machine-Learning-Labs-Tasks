# ============================================================
# MACHINE LEARNING LAB 05
# TASK 2 - POLYNOMIAL REGRESSION WITH MULTIPLE FEATURES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 1. CREATE A LARGE MULTI-FEATURE DATASET
# ============================================================

np.random.seed(42)

n = 150

experience = np.random.uniform(1, 20, n)
education = np.random.uniform(1, 10, n)
training = np.random.uniform(1, 15, n)

# Nonlinear relationship
salary = (
    20
    + 3 * experience
    + 2 * education
    + 1.5 * training
    + 0.35 * experience**2
    + 0.20 * education**2
    + 0.10 * training**2
    + 0.15 * experience * education
    + np.random.normal(0, 8, n)
)

dataset = pd.DataFrame({
    "Experience": experience,
    "Education": education,
    "Training_Hours": training,
    "Salary": salary
})


# ============================================================
# 2. DISPLAY DATASET INFORMATION
# ============================================================

print("=" * 70)
print("TASK 2 - POLYNOMIAL REGRESSION WITH MULTIPLE FEATURES")
print("=" * 70)

print("\nComplete Dataset:")
print(dataset)

print("\nFirst 10 Rows:")
print(dataset.head(10))

print("\nDataset Shape:")
print(dataset.shape)

print("\nDataset Information:")
dataset.info()

print("\nStatistical Summary:")
print(dataset.describe())


# ============================================================
# SCREENSHOT 1
# DATASET, SHAPE, INFO AND SUMMARY
# ============================================================


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(dataset.isnull().sum())


# ============================================================
# SCREENSHOT 2
# MISSING VALUES
# ============================================================


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = dataset[
    ["Experience", "Education", "Training_Hours"]
]

y = dataset["Salary"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())


# ============================================================
# 5. VISUALIZE NONLINEAR RELATIONSHIPS
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)

# Experience vs Salary
axes[0].scatter(
    dataset["Experience"],
    dataset["Salary"]
)

axes[0].set_title(
    "Experience vs Salary"
)

axes[0].set_xlabel(
    "Experience"
)

axes[0].set_ylabel(
    "Salary"
)

axes[0].grid(True)


# Education vs Salary
axes[1].scatter(
    dataset["Education"],
    dataset["Salary"]
)

axes[1].set_title(
    "Education vs Salary"
)

axes[1].set_xlabel(
    "Education"
)

axes[1].set_ylabel(
    "Salary"
)

axes[1].grid(True)


# Training Hours vs Salary
axes[2].scatter(
    dataset["Training_Hours"],
    dataset["Salary"]
)

axes[2].set_title(
    "Training Hours vs Salary"
)

axes[2].set_xlabel(
    "Training Hours"
)

axes[2].set_ylabel(
    "Salary"
)

axes[2].grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# SCREENSHOT 3
# FEATURE VS TARGET RELATIONSHIPS
# ============================================================


# ============================================================
# 6. TRAIN-TEST SPLIT
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
# TRAIN-TEST SPLIT
# ============================================================


# ============================================================
# 7. FUNCTION FOR POLYNOMIAL REGRESSION
# ============================================================

def train_polynomial_model(degree):

    poly = PolynomialFeatures(
        degree=degree
    )

    X_train_poly = poly.fit_transform(
        X_train
    )

    X_test_poly = poly.transform(
        X_test
    )

    model = LinearRegression()

    model.fit(
        X_train_poly,
        y_train
    )

    predictions = model.predict(
        X_test_poly
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return (
        poly,
        model,
        predictions,
        mse,
        r2
    )


# ============================================================
# 8. DEGREE 1
# ============================================================

poly1, model1, predictions1, mse1, r2_1 = (
    train_polynomial_model(1)
)

print("\n" + "=" * 50)
print("DEGREE 1 RESULTS")
print("=" * 50)

print("MSE:", mse1)
print("R2 Score:", r2_1)


# ============================================================
# SCREENSHOT 5
# DEGREE 1 RESULTS
# ============================================================


# ============================================================
# DEGREE 1 - ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    y_test,
    predictions1,
    s=40
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linewidth=2
)

plt.title(
    "Degree 1 - Actual vs Predicted"
)

plt.xlabel(
    "Actual Salary"
)

plt.ylabel(
    "Predicted Salary"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# SCREENSHOT 6
# DEGREE 1 GRAPH
# ============================================================


# ============================================================
# 9. DEGREE 2
# ============================================================

poly2, model2, predictions2, mse2, r2_2 = (
    train_polynomial_model(2)
)

print("\n" + "=" * 50)
print("DEGREE 2 RESULTS")
print("=" * 50)

print("MSE:", mse2)
print("R2 Score:", r2_2)


# ============================================================
# SCREENSHOT 7
# DEGREE 2 RESULTS
# ============================================================


# ============================================================
# DEGREE 2 - ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    y_test,
    predictions2,
    s=40
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linewidth=2
)

plt.title(
    "Degree 2 - Actual vs Predicted"
)

plt.xlabel(
    "Actual Salary"
)

plt.ylabel(
    "Predicted Salary"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# SCREENSHOT 8
# DEGREE 2 GRAPH
# ============================================================


# ============================================================
# 10. DEGREE 3
# ============================================================

poly3, model3, predictions3, mse3, r2_3 = (
    train_polynomial_model(3)
)

print("\n" + "=" * 50)
print("DEGREE 3 RESULTS")
print("=" * 50)

print("MSE:", mse3)
print("R2 Score:", r2_3)


# ============================================================
# SCREENSHOT 9
# DEGREE 3 RESULTS
# ============================================================


# ============================================================
# DEGREE 3 - ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    y_test,
    predictions3,
    s=40
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linewidth=2
)

plt.title(
    "Degree 3 - Actual vs Predicted"
)

plt.xlabel(
    "Actual Salary"
)

plt.ylabel(
    "Predicted Salary"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# SCREENSHOT 10
# DEGREE 3 GRAPH
# ============================================================


# ============================================================
# 11. DEGREE 4
# ============================================================

poly4, model4, predictions4, mse4, r2_4 = (
    train_polynomial_model(4)
)

print("\n" + "=" * 50)
print("DEGREE 4 RESULTS")
print("=" * 50)

print("MSE:", mse4)
print("R2 Score:", r2_4)


# ============================================================
# SCREENSHOT 11
# DEGREE 4 RESULTS
# ============================================================


# ============================================================
# DEGREE 4 - ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    y_test,
    predictions4,
    s=40
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linewidth=2
)

plt.title(
    "Degree 4 - Actual vs Predicted"
)

plt.xlabel(
    "Actual Salary"
)

plt.ylabel(
    "Predicted Salary"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# SCREENSHOT 12
# DEGREE 4 GRAPH
# ============================================================


# ============================================================
# 12. COMPARISON TABLE
# ============================================================

results = pd.DataFrame({
    "Degree": [1, 2, 3, 4],

    "MSE": [
        mse1,
        mse2,
        mse3,
        mse4
    ],

    "R2 Score": [
        r2_1,
        r2_2,
        r2_3,
        r2_4
    ]
})

print("\n" + "=" * 70)
print("POLYNOMIAL DEGREE COMPARISON")
print("=" * 70)

print(results)


# ============================================================
# SCREENSHOT 13
# DEGREE COMPARISON TABLE
# ============================================================


# ============================================================
# 13. MSE COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    results["Degree"],
    results["MSE"],
    marker="o",
    linewidth=2
)

plt.title(
    "MSE Comparison - Multiple Feature Polynomial Regression"
)

plt.xlabel(
    "Polynomial Degree"
)

plt.ylabel(
    "Mean Squared Error"
)

plt.xticks(
    [1, 2, 3, 4]
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# SCREENSHOT 14
# MSE COMPARISON GRAPH
# ============================================================


# ============================================================
# 14. R2 COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    results["Degree"],
    results["R2 Score"],
    marker="o",
    linewidth=2
)

plt.title(
    "R² Score Comparison - Multiple Feature Polynomial Regression"
)

plt.xlabel(
    "Polynomial Degree"
)

plt.ylabel(
    "R² Score"
)

plt.xticks(
    [1, 2, 3, 4]
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# SCREENSHOT 15
# R2 COMPARISON GRAPH
# ============================================================


# ============================================================
# 15. FIND BEST MODEL
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

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    "Best Polynomial Degree:",
    best_degree
)

print(
    "Best MSE:",
    best_mse
)

print(
    "Best R2 Score:",
    best_r2
)


# ============================================================
# SCREENSHOT 16
# BEST MODEL
# ============================================================


# ============================================================
# 16. FEATURE NAMES AFTER POLYNOMIAL TRANSFORMATION
# ============================================================

best_poly, best_model, best_predictions, _, _ = (
    train_polynomial_model(best_degree)
)

feature_names = best_poly.get_feature_names_out(
    X.columns
)

print("\nPolynomial Features Generated:")

print(feature_names)

print(
    "\nTotal Polynomial Features:",
    len(feature_names)
)


# ============================================================
# SCREENSHOT 17
# POLYNOMIAL FEATURES
# ============================================================


# ============================================================
# 17. UNDERFITTING VS GOOD FIT VS HIGH COMPLEXITY
# ============================================================

print("\n" + "=" * 70)
print("UNDERFITTING VS GOOD FIT VS HIGH COMPLEXITY")
print("=" * 70)


# Degree 1 - Underfitting
poly1_underfit = PolynomialFeatures(
    degree=1
)

X_train_poly1_underfit = (
    poly1_underfit.fit_transform(X_train)
)

X_test_poly1_underfit = (
    poly1_underfit.transform(X_test)
)

model1_underfit = LinearRegression()

model1_underfit.fit(
    X_train_poly1_underfit,
    y_train
)

prediction1_underfit = (
    model1_underfit.predict(
        X_test_poly1_underfit
    )
)

mse1_underfit = mean_squared_error(
    y_test,
    prediction1_underfit
)

r2_1_underfit = r2_score(
    y_test,
    prediction1_underfit
)


# Degree 2 - Good Fit
poly2_good = PolynomialFeatures(
    degree=2
)

X_train_poly2_good = (
    poly2_good.fit_transform(X_train)
)

X_test_poly2_good = (
    poly2_good.transform(X_test)
)

model2_good = LinearRegression()

model2_good.fit(
    X_train_poly2_good,
    y_train
)

prediction2_good = (
    model2_good.predict(
        X_test_poly2_good
    )
)

mse2_good = mean_squared_error(
    y_test,
    prediction2_good
)

r2_2_good = r2_score(
    y_test,
    prediction2_good
)


# Degree 4 - High Complexity / Possible Overfitting
poly4_complex = PolynomialFeatures(
    degree=4
)

X_train_poly4_complex = (
    poly4_complex.fit_transform(X_train)
)

X_test_poly4_complex = (
    poly4_complex.transform(X_test)
)

model4_complex = LinearRegression()

model4_complex.fit(
    X_train_poly4_complex,
    y_train
)

prediction4_complex = (
    model4_complex.predict(
        X_test_poly4_complex
    )
)

mse4_complex = mean_squared_error(
    y_test,
    prediction4_complex
)

r2_4_complex = r2_score(
    y_test,
    prediction4_complex
)


# ============================================================
# 18. PRINT UNDERFITTING / GOOD FIT / HIGH COMPLEXITY RESULTS
# ============================================================

print("\nDegree 1 - Underfitting")

print(
    "MSE:",
    mse1_underfit
)

print(
    "R2 Score:",
    r2_1_underfit
)


print("\nDegree 2 - Good Fit")

print(
    "MSE:",
    mse2_good
)

print(
    "R2 Score:",
    r2_2_good
)


print(
    "\nDegree 4 - High Complexity / Possible Overfitting"
)

print(
    "MSE:",
    mse4_complex
)

print(
    "R2 Score:",
    r2_4_complex
)


# ============================================================
# SCREENSHOT 18
# UNDERFITTING / GOOD FIT / HIGH COMPLEXITY RESULTS
# ============================================================


# ============================================================
# 19. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(11, 7))

plt.scatter(
    y_test,
    prediction1_underfit,
    s=40,
    label="Degree 1 - Underfitting"
)

plt.scatter(
    y_test,
    prediction2_good,
    s=40,
    label="Degree 2 - Good Fit"
)

plt.scatter(
    y_test,
    prediction4_complex,
    s=40,
    label="Degree 4 - High Complexity"
)


# Perfect prediction line
minimum = min(
    y_test.min(),
    prediction1_underfit.min(),
    prediction2_good.min(),
    prediction4_complex.min()
)

maximum = max(
    y_test.max(),
    prediction1_underfit.max(),
    prediction2_good.max(),
    prediction4_complex.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linewidth=2,
    label="Perfect Prediction"
)

plt.title(
    "Task 2 - Underfitting vs Good Fit vs High Complexity"
)

plt.xlabel(
    "Actual Salary"
)

plt.ylabel(
    "Predicted Salary"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# SCREENSHOT 19
# UNDERFITTING VS GOOD FIT VS HIGH COMPLEXITY GRAPH
# ============================================================


# ============================================================
# TASK 2 COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("TASK 2 COMPLETED SUCCESSFULLY")
print("=" * 70)