"""
Real-World Finance Project: Stock Price Analysis & Next-Day Direction Prediction
-----------------------------------------------------------------------------------
End-to-end analysis of daily stock price data: EDA, technical indicator
feature engineering, and a next-day price direction prediction model
(Logistic Regression vs Random Forest), evaluated honestly against a
naive baseline.

Author: Namra
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)


# --------------------------------------------------------------------------
# 1. Load / generate the dataset
# --------------------------------------------------------------------------
def generate_data(n_days=750, seed=42):
    """
    Simulates ~3 years of daily OHLCV data using Geometric Brownian Motion
    with volatility clustering (calm periods interrupted by turbulent
    stretches) — the same simulation approach used in quantitative finance.
    Replace this function with pd.read_csv("your_file.csv") (or a yfinance
    download) to use real data instead.
    """
    np.random.seed(seed)

    dates = pd.bdate_range(start="2023-01-02", periods=n_days)

    mu = 0.0004
    base_vol = 0.012
    returns = np.zeros(n_days)
    vol = np.zeros(n_days)
    vol[0] = base_vol

    for t in range(1, n_days):
        vol[t] = 0.90 * vol[t - 1] + 0.10 * base_vol + 0.05 * abs(returns[t - 1])
        shock = np.random.normal(0, 1)
        returns[t] = mu + vol[t] * shock

    close = 150 * np.exp(np.cumsum(returns))
    volume = np.round(np.random.gamma(5, 200000, n_days) * (1 + 3 * np.abs(returns))).astype(int)

    df = pd.DataFrame({"date": dates, "close": np.round(close, 2), "volume": volume})
    df["open"] = np.round(df["close"].shift(1).fillna(df["close"].iloc[0]) * (1 + np.random.normal(0, 0.002, n_days)), 2)
    df["high"] = np.round(df[["open", "close"]].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.004, n_days))), 2)
    df["low"] = np.round(df[["open", "close"]].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.004, n_days))), 2)
    df = df[["date", "open", "high", "low", "close", "volume"]]

    # realistic imperfections
    idx = np.random.choice(df.index, size=8, replace=False)
    df.loc[idx, "volume"] = np.nan
    df = pd.concat([df, df.sample(5, random_state=1)]).sort_index()

    return df


# --------------------------------------------------------------------------
# 2. Cleaning
# --------------------------------------------------------------------------
def clean_data(df):
    print("Duplicates:", df.duplicated(subset=["date"]).sum())
    print("Missing values:\n", df.isnull().sum())

    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["volume"] = df["volume"].fillna(df["volume"].median())
    return df


# --------------------------------------------------------------------------
# 3. Feature engineering
# --------------------------------------------------------------------------
def engineer_features(df):
    """Adds standard technical indicators and the prediction target."""
    df["daily_return"] = df["close"].pct_change()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_50"] = df["close"].rolling(50).mean()
    df["volatility_10"] = df["daily_return"].rolling(10).std()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df["lag_return_1"] = df["daily_return"].shift(1)
    df["lag_return_2"] = df["daily_return"].shift(2)

    # Target: did price go UP the next trading day?
    df["target_direction"] = (df["close"].shift(-1) > df["close"]).astype(int)

    return df


# --------------------------------------------------------------------------
# 4. EDA visualizations
# --------------------------------------------------------------------------
def plot_price_and_returns(df, save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(df["date"], df["close"], color="steelblue")
    axes[0].set_title("Closing Price Over Time")
    axes[0].set_ylabel("Price ($)")

    axes[1].plot(df["date"], df["daily_return"], color="indianred", linewidth=0.8)
    axes[1].set_title("Daily Returns Over Time (Volatility Clustering Visible)")
    axes[1].set_ylabel("Daily Return")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_distributions(df, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.histplot(df["daily_return"].dropna(), bins=40, kde=True, ax=axes[0], color="teal")
    axes[0].set_title("Distribution of Daily Returns")

    sns.histplot(df["volume"], bins=40, ax=axes[1], color="goldenrod")
    axes[1].set_title("Distribution of Trading Volume")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_moving_averages(df, save_path=None):
    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df["close"], label="Close", color="black", linewidth=1)
    plt.plot(df["date"], df["ma_10"], label="10-day MA", color="steelblue")
    plt.plot(df["date"], df["ma_50"], label="50-day MA", color="darkorange")
    plt.title("Price with Moving Averages")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# --------------------------------------------------------------------------
# 5. Modeling (time-aware split — never shuffle time series data)
# --------------------------------------------------------------------------
def time_aware_split(df_model, feature_cols, target_col="target_direction", test_frac=0.2):
    X = df_model[feature_cols]
    y = df_model[target_col]

    split_idx = int(len(df_model) * (1 - test_frac))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print("Train period:", df_model["date"].iloc[0].date(), "to", df_model["date"].iloc[split_idx - 1].date())
    print("Test period:", df_model["date"].iloc[split_idx].date(), "to", df_model["date"].iloc[-1].date())

    return X_train, X_test, y_train, y_test


def train_and_evaluate(X_train, X_test, y_train, y_test, seed=42):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=seed),
    }

    predictions, probabilities, results = {}, {}, []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        predictions[name] = preds
        probabilities[name] = probs
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "ROC-AUC": roc_auc_score(y_test, probs),
        })

    baseline_acc = max(y_test.mean(), 1 - y_test.mean())
    results.append({"Model": "Baseline (majority class)", "Accuracy": baseline_acc, "ROC-AUC": 0.5})

    results_df = pd.DataFrame(results).set_index("Model").round(3)
    return predictions, probabilities, results_df


def plot_confusion_matrices(predictions, y_test, save_path=None):
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, preds) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down", "Up"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curves(probabilities, y_test, save_path=None):
    plt.figure(figsize=(7, 6))
    for name, probs in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: Predicting Next-Day Direction")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# --------------------------------------------------------------------------
# 6. Main
# --------------------------------------------------------------------------
def main():
    print("Generating dataset...")
    raw_df = generate_data()
    raw_df.to_csv("stock_data.csv", index=False)

    df = clean_data(raw_df.copy())
    df = engineer_features(df)

    print("\nGenerating EDA visualizations...")
    plot_price_and_returns(df, save_path="price_returns.png")
    plot_distributions(df, save_path="distributions.png")
    plot_moving_averages(df, save_path="moving_averages.png")

    feature_cols = ["daily_return", "ma_10", "ma_50", "volatility_10",
                    "rsi_14", "lag_return_1", "lag_return_2", "volume"]
    df_model = df.dropna().reset_index(drop=True)

    X_train, X_test, y_train, y_test = time_aware_split(df_model, feature_cols)

    print("\nTraining models...")
    predictions, probabilities, results_df = train_and_evaluate(X_train, X_test, y_train, y_test)
    print("\nResults:\n", results_df)

    plot_confusion_matrices(predictions, y_test, save_path="confusion_matrices.png")
    plot_roc_curves(probabilities, y_test, save_path="roc_curves.png")

    print("\nConclusion: model accuracy is close to the naive baseline, which is the")
    print("expected result for short-term stock direction (consistent with the Efficient")
    print("Market Hypothesis) — this project demonstrates the ability to tell genuine")
    print("signal apart from noise, not just chase a high accuracy number.")


if __name__ == "__main__":
    main()
