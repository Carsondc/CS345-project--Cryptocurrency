

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import preprocess
import sys
os.makedirs("static/plots", exist_ok=True)
HORIZON = int(sys.argv[1])
POLY_DEGREES = [1, 2, 3, 4, 5, 6, 7, 8]
FEATURE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7]
file = sys.argv[2] 
files = []
files.append(file)
processor = preprocess.preprocess(files, HORIZON)
datasets = processor.generate_data()
X, y = datasets[0]
print("X shape:", X.shape)
print("y shape:", y.shape)
def naive_mae(y_true, X):
    preds = X[:len(y_true), 4]
    return mean_absolute_error(y_true, preds)

naive_MAE = naive_mae(y, X)
errors = {}
pipelines = {}
fig, axs = plt.subplots(4, 2, figsize=(16, 20))
axs = axs.flatten()
for i, deg in enumerate(POLY_DEGREES):
    X_train, X_test, y_train, y_test = train_test_split(
        X[:, FEATURE_INDICES],
        y,
        test_size=0.2,
        shuffle=False
    )
    pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=deg)),
        ("ridge", RidgeCV(alphas=np.logspace(-6, 6, 13)))
    ])

    pipeline.fit(X_train, y_train)
    pipelines[deg] = pipeline

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_MAE = mean_absolute_error(y_train, y_train_pred)
    test_MAE = mean_absolute_error(y_test, y_test_pred)
    train_RMSE = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_RMSE = np.sqrt(mean_squared_error(y_test, y_test_pred))

    errors[deg] = {
        "train_MAE": train_MAE,
        "test_MAE": test_MAE,
        "train_RMSE": train_RMSE,
        "test_RMSE": test_RMSE,
        "naive_MAE": naive_MAE
    }
    ax = axs[i]
    ax.plot(X_train[:, 0], y_train, "o", label="Train")
    ax.plot(X_test[:, 0], y_test, "o", label="Test")
    ax.plot(X_train[:, 0], y_train_pred, "x", label="Train Pred")
    ax.plot(X_test[:, 0], y_test_pred, "+", label="Test Pred")

    ax.set_title(f"Polynomial Degree {deg}")
    ax.set_xlabel("Days Since Start")
    ax.set_ylabel("Closing Price")
    ax.grid()
    ax.legend(fontsize=8)
    
fig.tight_layout()
fig.savefig("static/plots/best_fit_polynomials.png", dpi=300, bbox_inches="tight")
#plt.show()
df_errors = pd.DataFrame.from_dict(errors, orient="index").reset_index()
df_errors.rename(columns={"index": "degree"}, inplace=True)
df_errors = df_errors.sort_values(["test_RMSE", "test_MAE"])
best_row = df_errors.iloc[0]
best_degree = int(best_row["degree"])
best_pipeline = pipelines[best_degree]
latest_X = X[-1, FEATURE_INDICES].reshape(1, -1)
current_prediction = best_pipeline.predict(latest_X)[0]

print(f"\nPredicted closing price in {HORIZON} days: {float(current_prediction):.6f}")
with open("prediction.txt", "w") as f:
    f.write(f'{current_prediction:.6f}')
f.close()

del pipelines

print("\nModel comparison:")
print(df_errors.to_string(index=False))
print("\nBest model:")
print(best_row)
plt.figure(figsize=(10, 6))
plt.plot(df_errors["degree"], df_errors["test_RMSE"], marker="o", label="Test RMSE")
plt.axhline(y=naive_MAE, linestyle="--", color="red", label="Naive MAE")
plt.xlabel("Polynomial Degree")
plt.ylabel("Error")
plt.title("Model Error vs Naive Baseline")
plt.grid()
plt.legend()

plt.savefig("static/plots/error_vs_baseline.png", dpi=300, bbox_inches="tight")
#plt.show()
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

axs[0].bar(
    ["Train", "Test", "Naive"],
    [best_row["train_RMSE"], best_row["test_RMSE"], naive_MAE]
)
axs[0].set_title("RMSE Comparison")

axs[1].bar(
    ["Train", "Test", "Naive"],
    [best_row["train_MAE"], best_row["test_MAE"], naive_MAE]
)
axs[1].set_title("MAE Comparison")

for ax in axs:
    ax.grid(axis="y")

fig.tight_layout()
fig.savefig("static/plots/bar_chart_errors.png", dpi=300, bbox_inches="tight")
print("all figures saved")
#plt.show()


