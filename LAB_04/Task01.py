import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. CREATE / LOAD DATASET
# ============================================================


data = {
    "Hours": [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
        91, 92, 93, 94, 95, 96, 97, 98, 99, 100
    ],

    "Marks": [
        35, 40, 45, 50, 55, 60, 64, 69, 73, 78,
        81, 84, 88, 91, 94, 97, 100, 103, 106, 109,
        112, 115, 118, 121, 124, 127, 130, 133, 136, 139,
        141, 144, 147, 150, 152, 155, 158, 160, 163, 165,
        168, 170, 173, 175, 177, 180, 182, 184, 187, 189,
        191, 193, 195, 197, 199, 201, 203, 205, 207, 209,
        211, 213, 215, 216, 218, 220, 222, 223, 225, 227,
        228, 230, 232, 233, 235, 236, 238, 239, 241, 242,
        244, 245, 247, 248, 249, 251, 252, 253, 255, 256,
        257, 259, 260, 261, 262, 264, 265, 266, 267, 268
    ]
}

dataset = pd.DataFrame(data)

print("========== DATASET ==========")
print(dataset)
print("\nDataset Shape:", dataset.shape)
print("\nFirst 5 records:")
print(dataset.head())
print("\nLast 5 records:")
print(dataset.tail())
print("\nDataset Statistics:")
print(dataset.describe())



# ============================================================
# 2. DEFINE X AND Y
# ============================================================

X = dataset["Hours"].values
Y = dataset["Marks"].values

print("\n========== VARIABLES ==========")
print("Independent Variable (X): Hours")
print("Dependent Variable (Y): Marks")


# ============================================================
# 3. SPLIT DATA INTO TRAINING AND TESTING
# ============================================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

print("\n========== DATA SPLIT ==========")
print("X_train:", X_train)
print("X_test:", X_test)
print("Y_train:", Y_train)
print("Y_test:", Y_test)


# ============================================================
# 4. CALCULATE SLOPE (B1) AND INTERCEPT (B0) MANUALLY
# ============================================================

# Mean of X and Y
X_mean = np.mean(X_train)
Y_mean = np.mean(Y_train)

# Calculate slope
numerator = np.sum((X_train - X_mean) * (Y_train - Y_mean))
denominator = np.sum((X_train - X_mean) ** 2)

B1 = numerator / denominator

# Calculate intercept
B0 = Y_mean - (B1 * X_mean)


print("\n========== REGRESSION COEFFICIENTS ==========")
print("Slope (B1):", B1)
print("Intercept (B0):", B0)

print("\nRegression Equation:")
print("Y =", B0, "+", B1, "* X")


# ============================================================
# 5. PREDICTION FUNCTION
# ============================================================

def predict(X):
    return B0 + B1 * X


# Predictions
Y_train_pred = predict(X_train)
Y_test_pred = predict(X_test)


# ============================================================
# 6. DISPLAY PREDICTIONS
# ============================================================

print("\n========== TRAINING PREDICTIONS ==========")

for i in range(len(X_train)):
    print(
        "Hours:", X_train[i],
        "| Actual Marks:", Y_train[i],
        "| Predicted Marks:", round(Y_train_pred[i], 2)
    )


print("\n========== TESTING PREDICTIONS ==========")

for i in range(len(X_test)):
    print(
        "Hours:", X_test[i],
        "| Actual Marks:", Y_test[i],
        "| Predicted Marks:", round(Y_test_pred[i], 2)
    )


# ============================================================
# 7. TRAINING METRICS
# ============================================================

train_mae = mean_absolute_error(Y_train, Y_train_pred)
train_mse = mean_squared_error(Y_train, Y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(Y_train, Y_train_pred)

# SSE
train_sse = np.sum((Y_train - Y_train_pred) ** 2)


# ============================================================
# 8. TESTING METRICS
# ============================================================

test_mae = mean_absolute_error(Y_test, Y_test_pred)
test_mse = mean_squared_error(Y_test, Y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(Y_test, Y_test_pred)

# SSE
test_sse = np.sum((Y_test - Y_test_pred) ** 2)


# ============================================================
# 9. DISPLAY METRICS
# ============================================================

print("\n========================================")
print("          TRAINING METRICS")
print("========================================")

print("MAE  :", train_mae)
print("MSE  :", train_mse)
print("RMSE :", train_rmse)
print("SSE  :", train_sse)
print("R²   :", train_r2)


print("\n========================================")
print("           TESTING METRICS")
print("========================================")

print("MAE  :", test_mae)
print("MSE  :", test_mse)
print("RMSE :", test_rmse)
print("SSE  :", test_sse)
print("R²   :", test_r2)


# ============================================================
# 10. PREDICT NEW DATA
# ============================================================

new_hours = np.array([16, 17, 18])

new_predictions = predict(new_hours)

print("\n========== NEW PREDICTIONS ==========")

for i in range(len(new_hours)):
    print(
        "Hours:", new_hours[i],
        "| Predicted Marks:", round(new_predictions[i], 2)
    )


# ============================================================
# 11. TRAINING GRAPH
# ============================================================

# Sort values so regression line is drawn correctly
train_order = np.argsort(X_train)

plt.figure(figsize=(8, 5))

plt.scatter(
    X_train,
    Y_train,
    label="Training Data"
)

plt.plot(
    X_train[train_order],
    Y_train_pred[train_order],
    label="Regression Line"
)

plt.title("Simple Linear Regression - Training Data")
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 12. TESTING GRAPH
# ============================================================

test_order = np.argsort(X_test)

plt.figure(figsize=(8, 5))

plt.scatter(
    X_test,
    Y_test,
    label="Testing Data"
)

plt.plot(
    X_test[test_order],
    Y_test_pred[test_order],
    label="Regression Line"
)

plt.title("Simple Linear Regression - Testing Data")
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n========================================")
print("             FINAL SUMMARY")
print("========================================")

print("Regression Equation:")
print(f"Y = {B0:.2f} + {B1:.2f}X")

print("\nTraining R²:", round(train_r2, 4))
print("Testing R² :", round(test_r2, 4))

print("\nTraining RMSE:", round(train_rmse, 4))
print("Testing RMSE :", round(test_rmse, 4))

print("\nTASK COMPLETED!")