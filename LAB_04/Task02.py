# ============================================================
# LAB 4B - MULTIPLE LINEAR REGRESSION
# Student Performance Prediction
# ============================================================


# ============================================================
# SCREENSHOT 2 - IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.feature_selection import SelectKBest, f_regression


# ============================================================
# SCREENSHOT 1 - CREATE DATASET
# ============================================================

# Set random seed so the same random data is generated
# every time the program runs
np.random.seed(42)

# Number of students
n = 100

# Generate independent variables

# Study hours between 1 and 10
study_hours = np.random.randint(1, 11, n)

# Attendance between 50 and 100
attendance = np.random.randint(50, 101, n)

# Previous marks between 40 and 95
previous_marks = np.random.randint(40, 96, n)

# Generate dependent variable: Final Marks
# Final Marks depend on all three independent variables

final_marks = (
    2.0 * study_hours
    + 0.25 * attendance
    + 0.55 * previous_marks
    + np.random.normal(0, 4, n)
)

# Keep Final Marks between 0 and 100
final_marks = np.clip(
    final_marks,
    0,
    100
)


# Create Pandas DataFrame
dataset = pd.DataFrame({
    "Study_Hours": study_hours,
    "Attendance": attendance,
    "Previous_Marks": previous_marks,
    "Final_Marks": final_marks.round(2)
})


print("\n========== DATASET ==========")
print(dataset)


# ============================================================
# SCREENSHOT 3 - LOAD AND EXPLORE DATASET
# ============================================================

print("\n========== FIRST 5 RECORDS ==========")
print(dataset.head())


print("\n========== LAST 5 RECORDS ==========")
print(dataset.tail())


print("\n========== DATASET SHAPE ==========")
print(dataset.shape)


print("\n========== DATASET INFORMATION ==========")
dataset.info()


print("\n========== DATASET STATISTICS ==========")
print(dataset.describe())


# ============================================================
# SCREENSHOT 4A - CHECK CORRELATIONS
# ============================================================

print("\n========== CORRELATION MATRIX ==========")

correlation_matrix = dataset.corr()

print(correlation_matrix)


# ============================================================
# SCREENSHOT 4B - CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.show()


# ============================================================
# SCREENSHOT 5 - HANDLE MISSING VALUES
# AND ENCODE CATEGORICAL VARIABLES
# ============================================================

print("\n========== MISSING VALUES ==========")

print(dataset.isnull().sum())


# ------------------------------------------------------------
# Handle missing numerical values
# ------------------------------------------------------------

# Select all numerical columns
numeric_columns = dataset.select_dtypes(
    include=np.number
).columns


# If a numerical value is missing,
# replace it with the mean of that column
dataset[numeric_columns] = dataset[
    numeric_columns
].fillna(
    dataset[numeric_columns].mean()
)


print("\n========== MISSING VALUES AFTER HANDLING ==========")

print(dataset.isnull().sum())


# ------------------------------------------------------------
# Check categorical variables
# ------------------------------------------------------------

categorical_columns = dataset.select_dtypes(
    include=["object", "category"]
).columns


print("\n========== CATEGORICAL COLUMNS ==========")

print(list(categorical_columns))


# Perform One-Hot Encoding only if
# categorical variables exist
if len(categorical_columns) > 0:

    dataset = pd.get_dummies(
        dataset,
        columns=categorical_columns,
        drop_first=True
    )

    print(
        "\nCategorical variables encoded successfully."
    )

else:

    print(
        "\nNo categorical variables found."
        "\nOne-Hot Encoding is not required."
    )


# ============================================================
# SCREENSHOT 6 - DEFINE X AND y
# ============================================================

# Independent variables / predictors
X = dataset[
    [
        "Study_Hours",
        "Attendance",
        "Previous_Marks"
    ]
]


# Dependent / target variable
y = dataset["Final_Marks"]


print("\n========== INDEPENDENT VARIABLES X ==========")

print(X.head())


print("\n========== DEPENDENT VARIABLE y ==========")

print(y.head())


print("\nX Shape:", X.shape)

print("y Shape:", y.shape)


# ============================================================
# SCREENSHOT 7 - SPLIT INTO TRAIN AND TEST SETS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\n========== TRAIN / TEST SPLIT ==========")


print(
    "X_train Shape:",
    X_train.shape
)

print(
    "X_test Shape :",
    X_test.shape
)

print(
    "y_train Shape:",
    y_train.shape
)

print(
    "y_test Shape :",
    y_test.shape
)


# ============================================================
# SCREENSHOT 8 - TRAIN THE MODEL
# ============================================================

# Create Multiple Linear Regression model
model = LinearRegression()


# Train the model
model.fit(
    X_train,
    y_train
)


print("\n========== MODEL TRAINING ==========")

print(
    "Multiple Linear Regression model "
    "has been trained successfully."
)


# ============================================================
# SCREENSHOT 9 - PREDICT AND EVALUATE
# ============================================================

# Predict Final Marks for testing data
y_pred = model.predict(
    X_test
)


# ------------------------------------------------------------
# Mean Absolute Error
# ------------------------------------------------------------

MAE = mean_absolute_error(
    y_test,
    y_pred
)


# ------------------------------------------------------------
# Mean Squared Error
# ------------------------------------------------------------

MSE = mean_squared_error(
    y_test,
    y_pred
)


# ------------------------------------------------------------
# Root Mean Squared Error
# ------------------------------------------------------------

RMSE = np.sqrt(
    MSE
)


# ------------------------------------------------------------
# R-Squared
# ------------------------------------------------------------

R2 = r2_score(
    y_test,
    y_pred
)


# ------------------------------------------------------------
# Training R-Squared
# ------------------------------------------------------------

training_score = model.score(
    X_train,
    y_train
)


# ------------------------------------------------------------
# Testing R-Squared
# ------------------------------------------------------------

testing_score = model.score(
    X_test,
    y_test
)


print("\n========== MODEL PERFORMANCE ==========")


print(
    "Mean Absolute Error (MAE):",
    round(MAE, 4)
)


print(
    "Mean Squared Error (MSE):",
    round(MSE, 4)
)


print(
    "Root Mean Squared Error (RMSE):",
    round(RMSE, 4)
)


print(
    "R-Squared (R²):",
    round(R2, 4)
)


print(
    "Training R² Score:",
    round(training_score, 4)
)


print(
    "Testing R² Score:",
    round(testing_score, 4)
)


# ============================================================
# SCREENSHOT 10 - CREATE COMPARISON DATAFRAME
# ============================================================

comparison_df = pd.DataFrame({

    "Actual Value":
        y_test.values,

    "Predicted Value":
        y_pred,

    "Difference":
        y_test.values - y_pred
})


# Round values for clean output
comparison_df = comparison_df.round(2)


print("\n========== ACTUAL VS PREDICTED ==========")

print(comparison_df)


# ============================================================
# SCREENSHOT 11A
# ACTUAL VS PREDICTED SCATTER PLOT
# ============================================================

plt.figure(figsize=(8, 6))


plt.scatter(
    y_test,
    y_pred
)


# Find minimum value for reference line
minimum = min(
    y_test.min(),
    y_pred.min()
)


# Find maximum value for reference line
maximum = max(
    y_test.max(),
    y_pred.max()
)


# Perfect prediction reference line
plt.plot(
    [minimum, maximum],
    [minimum, maximum]
)


plt.xlabel(
    "Actual Final Marks"
)

plt.ylabel(
    "Predicted Final Marks"
)


plt.title(
    "Actual vs Predicted Final Marks"
)


plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# SCREENSHOT 11B
# STUDY HOURS VS FINAL MARKS
# Independent Variable 1 vs Dependent Variable
# ============================================================

plt.figure(figsize=(8, 6))


sns.regplot(
    x="Study_Hours",
    y="Final_Marks",
    data=dataset,
    ci=None
)


plt.xlabel(
    "Study Hours"
)

plt.ylabel(
    "Final Marks"
)


plt.title(
    "Study Hours vs Final Marks"
)


plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# SCREENSHOT 11C
# ATTENDANCE VS FINAL MARKS
# Independent Variable 2 vs Dependent Variable
# ============================================================

plt.figure(figsize=(8, 6))


sns.regplot(
    x="Attendance",
    y="Final_Marks",
    data=dataset,
    ci=None
)


plt.xlabel(
    "Attendance"
)

plt.ylabel(
    "Final Marks"
)


plt.title(
    "Attendance vs Final Marks"
)


plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# SCREENSHOT 11D
# PREVIOUS MARKS VS FINAL MARKS
# Independent Variable 3 vs Dependent Variable
# ============================================================

plt.figure(figsize=(8, 6))


sns.regplot(
    x="Previous_Marks",
    y="Final_Marks",
    data=dataset,
    ci=None
)


plt.xlabel(
    "Previous Marks"
)

plt.ylabel(
    "Final Marks"
)


plt.title(
    "Previous Marks vs Final Marks"
)


plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# SCREENSHOT 11E - RESIDUAL PLOT
# ============================================================

# Residual = Actual Value - Predicted Value
residuals = (
    y_test.values - y_pred
)


plt.figure(figsize=(8, 6))


plt.scatter(
    y_pred,
    residuals
)


# Zero residual reference line
plt.axhline(
    y=0,
    linestyle="--"
)


plt.xlabel(
    "Predicted Final Marks"
)

plt.ylabel(
    "Residuals"
)


plt.title(
    "Residual Plot"
)


plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# SCREENSHOT 12 - INTERPRET COEFFICIENTS
# ============================================================

print("\n========== MODEL INTERCEPT ==========")


print(
    "Intercept:",
    round(model.intercept_, 4)
)


# Create coefficient DataFrame
coefficient_df = pd.DataFrame({

    "Independent Variable":
        X.columns,

    "Coefficient":
        model.coef_
})


# Round coefficients
coefficient_df["Coefficient"] = (
    coefficient_df["Coefficient"].round(4)
)


print("\n========== MODEL COEFFICIENTS ==========")

print(coefficient_df)


print(
    "\n========== COEFFICIENT INTERPRETATION =========="
)


for feature, coefficient in zip(
    X.columns,
    model.coef_
):

    print(
        "\n",
        feature,
        "Coefficient =",
        round(coefficient, 4)
    )


    if coefficient > 0:

        print(
            "Increasing",
            feature,
            "by one unit increases the predicted "
            "Final Marks by approximately",
            round(coefficient, 4),
            "units, while other variables "
            "are held constant."
        )


    elif coefficient < 0:

        print(
            "Increasing",
            feature,
            "by one unit decreases the predicted "
            "Final Marks by approximately",
            abs(round(coefficient, 4)),
            "units, while other variables "
            "are held constant."
        )


    else:

        print(
            feature,
            "has no linear contribution "
            "to the prediction."
        )


# ============================================================
# SCREENSHOT 13 - FEATURE SELECTION
# ============================================================

# Select all features and calculate
# their F-regression scores
selector = SelectKBest(
    score_func=f_regression,
    k="all"
)


selector.fit(
    X,
    y
)


# Create feature selection table
feature_selection_df = pd.DataFrame({

    "Feature":
        X.columns,

    "F-Score":
        selector.scores_,

    "P-Value":
        selector.pvalues_
})


# Sort features from highest F-score
# to lowest F-score
feature_selection_df = (
    feature_selection_df
    .sort_values(
        by="F-Score",
        ascending=False
    )
)


# Round values for cleaner output
feature_selection_df["F-Score"] = (
    feature_selection_df["F-Score"].round(4)
)


print("\n========== FEATURE SELECTION ==========")

print(feature_selection_df)


print(
    "\nMost influential feature according "
    "to the F-regression test:"
)


print(
    feature_selection_df.iloc[0]["Feature"]
)


# ============================================================
# ADDITIONAL COEFFICIENT ANALYSIS
# ============================================================

coefficient_importance = pd.DataFrame({

    "Feature":
        X.columns,

    "Coefficient":
        model.coef_,

    "Absolute Coefficient":
        np.abs(model.coef_)
})


coefficient_importance = (
    coefficient_importance
    .sort_values(
        by="Absolute Coefficient",
        ascending=False
    )
)


coefficient_importance[
    "Coefficient"
] = coefficient_importance[
    "Coefficient"
].round(4)


coefficient_importance[
    "Absolute Coefficient"
] = coefficient_importance[
    "Absolute Coefficient"
].round(4)


print(
    "\n========== COEFFICIENT MAGNITUDES =========="
)


print(
    coefficient_importance
)


# ============================================================
# SCREENSHOT 14 - PREDICTION ON NEW DATA
# ============================================================

# Create a new student's data
new_student = pd.DataFrame({

    "Study_Hours": [7],

    "Attendance": [85],

    "Previous_Marks": [75]
})


# Predict Final Marks
new_prediction = model.predict(
    new_student
)


print("\n========== NEW DATA PREDICTION ==========")


print(
    "Study Hours     : 7"
)

print(
    "Attendance      : 85"
)

print(
    "Previous Marks  : 75"
)


print(
    "Predicted Final Marks:",
    round(
        new_prediction[0],
        2
    )
)


# ============================================================
# END
# ============================================================

print(
    "\nMultiple Linear Regression "
    "Lab Completed Successfully."
)