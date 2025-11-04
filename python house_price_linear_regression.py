import os
import sys
import math
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Configuration / Paths
# -------------------------------
DATA_FILENAME = "train.csv"           # standard Kaggle train file
MODEL_FILENAME = "linear_model.pkl"
SCALER_FILENAME = "scaler.pkl"
IMPUTER_FILENAME = "imputer.pkl"
PLOTS_DIR = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

# -------------------------------
# Helpers
# -------------------------------
def load_data(path):
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Download train.csv from Kaggle and place it here.")
        sys.exit(1)
    df = pd.read_csv(path)
    return df

def prepare_features(df):
    """
    Create simplified features:
      - SqFt: GrLivArea (above ground living area). (optionally + basement)
      - Bathrooms: FullBath + 0.5 * HalfBath
      - Bedrooms: BedroomAbvGr
    Also returns X, y.
    """
    # Copy to avoid SettingWithCopy warnings
    data = df.copy()

    # Create SqFt (you can uncomment the basement addition if you want)
    data["SqFt"] = data["GrLivArea"]  # + data["TotalBsmtSF"].fillna(0)

    # Bathrooms: treat half baths as 0.5
    data["Bathrooms"] = data["FullBath"].fillna(0) + 0.5 * data["HalfBath"].fillna(0)

    # Bedrooms
    data["Bedrooms"] = data["BedroomAbvGr"].fillna(0)

    # Target
    data["SalePrice"] = data["SalePrice"]

    # Select features we want
    features = ["SqFt", "Bedrooms", "Bathrooms"]
    target = "SalePrice"

    X = data[features]
    y = data[target]

    return X, y

def impute_and_scale(X_train, X_test):
    """
    Impute missing values (median) and scale features (StandardScaler).
    Returns processed arrays and persisted imputer/scaler objects.
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    return X_train_scaled, X_test_scaled, imputer, scaler

def plot_true_vs_pred(y_test, y_pred, outpath):
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    maxv = max(max(y_test.max(), y_pred.max()), 1)
    plt.plot([0, maxv], [0, maxv], linestyle='--')
    plt.xlabel("True SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title("True vs Predicted Sale Price")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_residuals_hist(y_test, residuals, outpath):
    plt.figure(figsize=(6,4))
    plt.hist(residuals, bins=50)
    plt.xlabel("Residual (Predicted - True)")
    plt.ylabel("Count")
    plt.title("Residuals Distribution")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_sqft_vs_price(X_raw, y_raw, outpath):
    """
    Scatter plot of SqFt vs SalePrice to visualize relationship.
    X_raw expected as DataFrame (unscaled) and contains SqFt column.
    """
    if "SqFt" not in X_raw.columns:
        return
    plt.figure(figsize=(6,4))
    plt.scatter(X_raw["SqFt"], y_raw, alpha=0.5)
    plt.xlabel("SqFt (GrLivArea)")
    plt.ylabel("SalePrice")
    plt.title("SqFt vs SalePrice")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# -------------------------------
# Main flow
# -------------------------------
def main():
    print("Loading data...")
    df = load_data(DATA_FILENAME)

    print("Preparing features...")
    X, y = prepare_features(df)

    # Quick info
    print("\nSample data (first 5 rows):")
    print(pd.concat([X, y], axis=1).head())

    # We'll hold out 20% for testing
    print("\nSplitting train/test (80/20)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Impute missing values and scale
    print("Imputing & scaling features...")
    X_train_scaled, X_test_scaled, imputer, scaler = impute_and_scale(X_train_raw, X_test_raw)

    # Save imputer & scaler for future use
    joblib.dump(imputer, IMPUTER_FILENAME)
    joblib.dump(scaler, SCALER_FILENAME)
    print(f"Saved imputer -> {IMPUTER_FILENAME}, scaler -> {SCALER_FILENAME}")

    # Train linear regression
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Save model
    joblib.dump(model, MODEL_FILENAME)
    print(f"Saved trained model -> {MODEL_FILENAME}\n")

    # Evaluate
    print("Evaluating on test set...")
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Test RMSE: {rmse:,.2f}")
    print(f"Test R^2: {r2:.4f}")

    # Coefficients relative to scaled features:
    coef = model.coef_
    intercept = model.intercept_
    print("\nModel coefficients (for scaled features):")
    for feat, c in zip(X.columns, coef):
        print(f"  {feat}: {c:.6f}")
    print(f"Intercept: {intercept:.6f}")

    # If you want coefficients on original feature scale (approximate interpretation),
    # we can transform them back: coef_orig = coef / scaler.scale_
    coef_orig = coef / scaler.scale_
    intercept_orig = intercept - np.sum((scaler.mean_ / scaler.scale_) * coef)
    print("\nApproximate coefficients on original feature scale (per unit change):")
    for feat, c in zip(X.columns, coef_orig):
        print(f"  {feat}: {c:.2f}")
    print(f"Approx intercept (on original SalePrice scale): {intercept_orig:.2f}")

    # Diagnostics & plots
    print("\nCreating diagnostic plots...")
    plot_true_vs_pred(y_test, y_pred, os.path.join(PLOTS_DIR, "true_vs_predicted.png"))
    residuals = y_pred - y_test
    plot_residuals_hist(y_test, residuals, os.path.join(PLOTS_DIR, "residuals_hist.png"))
    plot_sqft_vs_price(X, y, os.path.join(PLOTS_DIR, "sqft_vs_price.png"))

    print(f"Saved plots to {PLOTS_DIR}/ (true_vs_predicted.png, residuals_hist.png, sqft_vs_price.png)")

    # Print a quick top errors table
    errors_df = pd.DataFrame({
        "True": y_test,
        "Predicted": y_pred,
        "AbsError": np.abs(y_test - y_pred),
        "SqFt": X_test_raw["SqFt"].values,
        "Bedrooms": X_test_raw["Bedrooms"].values,
        "Bathrooms": X_test_raw["Bathrooms"].values
    }).sort_values("AbsError", ascending=False).head(10)

    print("\nTop 10 largest absolute errors (test set):")
    print(errors_df.to_string(index=False))

    print("\nDone.")

if __name__ == "__main__":
    main()