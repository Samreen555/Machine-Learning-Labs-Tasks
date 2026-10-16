# ============================================================
# BAHRIA UNIVERSITY, ISLAMABAD CAMPUS
# OPEN-ENDED LAB I
# Course: AIC 301 - Machine Learning Lab
# Title: Exploring Predictive Analysis with Regression
#
# Dataset: Energy Efficiency Dataset
# Target Variable: Heating_Load
#
# Models:
# 1. Linear Regression
# 2. Polynomial Regression (Degree 2)
# 3. Polynomial Regression (Degree 3)
# ============================================================


# ============================================================
# SECTION 1: IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


# Set visualization style
sns.set_theme(style="whitegrid")

print("=" * 60)
print("EXPLORING PREDICTIVE ANALYSIS WITH REGRESSION")
print("=" * 60)


# ============================================================
# SECTION 2: LOAD THE DATASET
# ============================================================

file_path = "Energy.xlsx"

df = pd.read_excel(file_path)

# Rename columns using meaningful feature names
df.columns = [
    "Relative_Compactness",
    "Surface_Area",
    "Wall_Area",
    "Roof_Area",
    "Overall_Height",
    "Orientation",
    "Glazing_Area",
    "Glazing_Area_Distribution",
    "Heating_Load",
    "Cooling_Load"
]

print("\nFIRST FIVE RECORDS")
print(df.head().to_string(index=False))

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())


# ============================================================
# SECTION 3: UNDERSTAND THE DATASET
# ============================================================

print("\nDATASET INFORMATION")
df.info()

print("\nSTATISTICAL SUMMARY")
print(df.describe().round(2))


# ============================================================
# SECTION 4: CHECK MISSING VALUES AND DUPLICATES
# ============================================================

print("\nMISSING VALUES IN EACH COLUMN")
print(df.isnull().sum())

print("\nTOTAL MISSING VALUES")
print(df.isnull().sum().sum())

print("\nTOTAL DUPLICATE RECORDS")
print(df.duplicated().sum())

# Check data types
print("\nDATA TYPES")
print(df.dtypes)


# ============================================================
# SECTION 5: DEFINE INPUT FEATURES AND TARGET VARIABLE
# ============================================================

# Eight building characteristics are used as input features.
# Cooling_Load is not used as an input because Heating_Load
# is the target variable for this experiment.

feature_columns = [
    "Relative_Compactness",
    "Surface_Area",
    "Wall_Area",
    "Roof_Area",
    "Overall_Height",
    "Orientation",
    "Glazing_Area",
    "Glazing_Area_Distribution"
]

X = df[feature_columns]
y = df["Heating_Load"]

print("\nINPUT FEATURES")
print(X.head().to_string(index=False))

print("\nTARGET VARIABLE: HEATING LOAD")
print(y.head().to_string(index=False))

print("\nNUMBER OF INPUT FEATURES:", X.shape[1])
print("NUMBER OF TARGET RECORDS:", len(y))


# ============================================================
# SECTION 6: ANALYZE THE TARGET VARIABLE
# ============================================================

print("\nHEATING LOAD STATISTICS")
print(y.describe().round(2))

print("\nMINIMUM HEATING LOAD:", round(y.min(), 2))
print("MAXIMUM HEATING LOAD:", round(y.max(), 2))
print("AVERAGE HEATING LOAD:", round(y.mean(), 2))
print("MEDIAN HEATING LOAD:", round(y.median(), 2))


# ============================================================
# SECTION 7: DISTRIBUTION OF HEATING LOAD
# ============================================================

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="Heating_Load",
    bins=25,
    kde=True
)

plt.title("Distribution of Heating Load")
plt.xlabel("Heating Load")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# ============================================================
# SECTION 8: BOXPLOT TO INVESTIGATE POTENTIAL OUTLIERS
# ============================================================

plt.figure(figsize=(9, 4))

sns.boxplot(
    x=df["Heating_Load"]
)

plt.title("Boxplot of Heating Load")
plt.xlabel("Heating Load")
plt.tight_layout()
plt.show()


# ============================================================
# SECTION 9: BOXPLOTS OF ALL NUMERICAL FEATURES
# ============================================================

plt.figure(figsize=(14, 7))

df[feature_columns].boxplot(rot=45)

plt.title("Boxplots of Building Features")
plt.ylabel("Feature Values")
plt.tight_layout()
plt.show()


# ============================================================
# SECTION 10: CORRELATION HEATMAP
# ============================================================

correlation_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Energy Efficiency Dataset")
plt.tight_layout()
plt.show()


# ============================================================
# SECTION 11: RELATIONSHIP BETWEEN FEATURES AND HEATING LOAD
# ============================================================

selected_features = [
    "Relative_Compactness",
    "Surface_Area",
    "Wall_Area",
    "Roof_Area",
    "Overall_Height",
    "Orientation",
    "Glazing_Area",
    "Glazing_Area_Distribution"
]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

for i, feature in enumerate(selected_features):
    row = i // 4
    col = i % 4

    sns.scatterplot(
        data=df,
        x=feature,
        y="Heating_Load",
        ax=axes[row, col]
    )

    axes[row, col].set_title(f"{feature} vs Heating Load")
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel("Heating Load")

plt.suptitle(
    "Relationships Between Building Features and Heating Load",
    fontsize=15
)

plt.tight_layout()
plt.show()


# ============================================================
# SECTION 12: PREPARE TRAINING AND TESTING DATA
# ============================================================

# 80% of data for training and 20% for testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTRAINING AND TESTING DATA")
print("Original dataset size:", X.shape)
print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)


# ============================================================
# SECTION 13: CREATE REGRESSION MODELS
# ============================================================

# MODEL 1: LINEAR REGRESSION

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)


# MODEL 2: POLYNOMIAL REGRESSION - DEGREE 2

polynomial_model_2 = Pipeline([
    (
        "polynomial_features",
        PolynomialFeatures(degree=2, include_bias=False)
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "regression",
        LinearRegression()
    )
])

polynomial_model_2.fit(X_train, y_train)


# MODEL 3: POLYNOMIAL REGRESSION - DEGREE 3

polynomial_model_3 = Pipeline([
    (
        "polynomial_features",
        PolynomialFeatures(degree=3, include_bias=False)
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "regression",
        LinearRegression()
    )
])

polynomial_model_3.fit(X_train, y_train)


print("\nALL THREE MODELS HAVE BEEN TRAINED SUCCESSFULLY.")


# ============================================================
# SECTION 14: MAKE PREDICTIONS
# ============================================================

# Predictions on training data
linear_train_predictions = linear_model.predict(X_train)
poly2_train_predictions = polynomial_model_2.predict(X_train)
poly3_train_predictions = polynomial_model_3.predict(X_train)

# Predictions on testing data
linear_test_predictions = linear_model.predict(X_test)
poly2_test_predictions = polynomial_model_2.predict(X_test)
poly3_test_predictions = polynomial_model_3.predict(X_test)


# ============================================================
# SECTION 15: EVALUATE MODEL PERFORMANCE
# ============================================================

models = {
    "Linear Regression": (
        linear_train_predictions,
        linear_test_predictions
    ),

    "Polynomial Regression (Degree 2)": (
        poly2_train_predictions,
        poly2_test_predictions
    ),

    "Polynomial Regression (Degree 3)": (
        poly3_train_predictions,
        poly3_test_predictions
    )
}

results = []

for model_name, predictions in models.items():

    train_predictions, test_predictions = predictions

    # Training metrics
    train_mse = mean_squared_error(
        y_train,
        train_predictions
    )

    train_r2 = r2_score(
        y_train,
        train_predictions
    )

    # Testing metrics
    test_mse = mean_squared_error(
        y_test,
        test_predictions
    )

    test_r2 = r2_score(
        y_test,
        test_predictions
    )

    results.append({
        "Model": model_name,
        "Train MSE": train_mse,
        "Test MSE": test_mse,
        "Train R2": train_r2,
        "Test R2": test_r2
    })


# Convert results into a DataFrame
results_df = pd.DataFrame(results)

print("\nMODEL PERFORMANCE COMPARISON")
print(
    results_df.to_string(
        index=False,
        formatters={
            "Train MSE": "{:.4f}".format,
            "Test MSE": "{:.4f}".format,
            "Train R2": "{:.4f}".format,
            "Test R2": "{:.4f}".format
        }
    )
)


# ============================================================
# SECTION 16: VISUALIZE MODEL PERFORMANCE
# ============================================================

# Compare test MSE

plt.figure(figsize=(10, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="Test MSE"
)

plt.title("Comparison of Regression Models Using Test MSE")
plt.xlabel("Regression Model")
plt.ylabel("Mean Squared Error")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# Compare test R-squared

plt.figure(figsize=(10, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="Test R2"
)

plt.title("Comparison of Regression Models Using Test R²")
plt.xlabel("Regression Model")
plt.ylabel("R-squared Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# ============================================================
# SECTION 17: ACTUAL VS PREDICTED VALUES
# ============================================================

predictions_dictionary = {
    "Linear Regression": linear_test_predictions,
    "Polynomial Regression (Degree 2)": poly2_test_predictions,
    "Polynomial Regression (Degree 3)": poly3_test_predictions
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (model_name, predictions) in zip(
    axes,
    predictions_dictionary.items()
):

    ax.scatter(
        y_test,
        predictions,
        alpha=0.7
    )

    # Reference line for perfect predictions
    min_value = min(y_test.min(), predictions.min())
    max_value = max(y_test.max(), predictions.max())

    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    ax.set_title(model_name)
    ax.set_xlabel("Actual Heating Load")
    ax.set_ylabel("Predicted Heating Load")

plt.suptitle("Actual vs Predicted Heating Load")
plt.tight_layout()
plt.show()




# ============================================================
# SECTION 18: RESIDUAL ANALYSIS
# ============================================================

# Residual = Actual value - Predicted value

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (model_name, predictions) in zip(
    axes,
    predictions_dictionary.items()
):

    residuals = y_test.to_numpy() - predictions

    ax.scatter(
        predictions,
        residuals,
        alpha=0.7
    )

    ax.axhline(
        y=0,
        linestyle="--"
    )

    ax.set_title(model_name)
    ax.set_xlabel("Predicted Heating Load")
    ax.set_ylabel("Residuals")

plt.suptitle("Residual Analysis of Regression Models")
plt.tight_layout()
plt.show()


# ============================================================
# SECTION 19: CHECK FOR POSSIBLE OVERFITTING
# ============================================================

print("\nOVERFITTING AND UNDERFITTING CHECK")

for _, row in results_df.iterrows():

    model_name = row["Model"]

    train_r2 = row["Train R2"]
    test_r2 = row["Test R2"]

    difference = train_r2 - test_r2

    print("\nModel:", model_name)
    print("Training R²:", round(train_r2, 4))
    print("Testing R²:", round(test_r2, 4))
    print("Train-Test R² Difference:", round(difference, 4))

    if train_r2 < 0.50 and test_r2 < 0.50:
        print(
            "Interpretation: Both scores are relatively low; "
            "the model may be underfitting."
        )

    elif difference > 0.10:
        print(
            "Interpretation: The model may be overfitting "
            "because training performance is noticeably better."
        )

    else:
        print(
            "Interpretation: No large train-test R² gap "
            "was detected using this simple check."
        )


# ============================================================
# SECTION 20: SELECT THE BEST MODEL
# ============================================================

# Choose the model with the lowest test MSE
best_model_row = results_df.loc[
    results_df["Test MSE"].idxmin()
]

best_model_name = best_model_row["Model"]

print("\nBEST MODEL BASED ON TEST MSE")
print("Model:", best_model_name)
print("Test MSE:", round(best_model_row["Test MSE"], 4))
print("Test R²:", round(best_model_row["Test R2"], 4))


# ============================================================
# SECTION 21: DISPLAY SAMPLE PREDICTIONS
# ============================================================

# Select the prediction array for the best model

best_predictions = predictions_dictionary[best_model_name]

comparison_df = pd.DataFrame({
    "Actual_Heating_Load": y_test.to_numpy(),
    "Predicted_Heating_Load": best_predictions
})

comparison_df["Absolute_Error"] = abs(
    comparison_df["Actual_Heating_Load"]
    - comparison_df["Predicted_Heating_Load"]
)

print("\nSAMPLE ACTUAL VS PREDICTED VALUES")
print(comparison_df.head(10).round(3).to_string(index=False))


# ============================================================
# SECTION 22: PREDICT HEATING LOAD FOR A NEW BUILDING
# ============================================================

# Example input values.
# Replace these with the characteristics of a building
# you want to evaluate.

new_building = pd.DataFrame([{
    "Relative_Compactness": 0.90,
    "Surface_Area": 563.5,
    "Wall_Area": 318.5,
    "Roof_Area": 122.5,
    "Overall_Height": 7.0,
    "Orientation": 2,
    "Glazing_Area": 0.10,
    "Glazing_Area_Distribution": 1
}])

# Ensure the input columns are in the correct order
new_building = new_building[feature_columns]

# Use the selected best model
if best_model_name == "Linear Regression":
    selected_model = linear_model

elif best_model_name == "Polynomial Regression (Degree 2)":
    selected_model = polynomial_model_2

else:
    selected_model = polynomial_model_3

predicted_heating_load = selected_model.predict(new_building)[0]

print("\nNEW BUILDING PREDICTION")
print(
    "Predicted Heating Load:",
    round(predicted_heating_load, 3)
)


# ============================================================
# SECTION 23: FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print("Dataset: Energy Efficiency Dataset")
print("Total observations:", len(df))
print("Number of input features:", len(feature_columns))
print("Target variable: Heating_Load")
print("Models evaluated:", len(results_df))
print("Best model based on test MSE:", best_model_name)

print(
    "Best model test MSE:",
    round(best_model_row["Test MSE"], 4)
)

print(
    "Best model test R²:",
    round(best_model_row["Test R2"], 4)
)



print("\nANALYSIS COMPLETED SUCCESSFULLY.")

