"""
Data Cleaning & Visualization Project
--------------------------------------
Cleans a raw movies dataset (missing values, duplicates, outliers)
and generates a set of visual insights using Pandas, Matplotlib, and Seaborn.
 
Author: Namra
"""
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
 
 
# --------------------------------------------------------------------------
# 1. Load / generate the raw dataset
# --------------------------------------------------------------------------
def generate_raw_data(n=500, seed=42):
    """
    Generates a synthetic 'messy' movies dataset with realistic data-quality
    issues: inconsistent text formatting, missing values, outliers, and
    duplicate rows. Replace this function with pd.read_csv("your_file.csv")
    to use a real dataset instead.
    """
    np.random.seed(seed)
 
    genres_clean = ["Action", "Comedy", "Drama", "Horror",
                    "Romance", "Sci-Fi", "Thriller", "Animation"]
    genres_messy = [
        np.random.choice([g, g.lower(), g.upper(), f" {g} ", g + "  "])
        for g in np.random.choice(genres_clean, n)
    ]
    directors = [f"Director_{i}" for i in range(1, 41)]
 
    df = pd.DataFrame({
        "movie_id": range(1, n + 1),
        "title": [f"Movie_{i}" for i in range(1, n + 1)],
        "genre": genres_messy,
        "director": np.random.choice(directors, n),
        "release_year": np.random.randint(1980, 2025, n),
        "budget_musd": np.round(np.random.gamma(4, 15, n), 1),
        "revenue_musd": np.round(np.random.gamma(5, 25, n), 1),
        "rating": np.round(np.clip(np.random.normal(6.5, 1.2, n), 1, 10), 1),
        "runtime_min": np.random.randint(80, 180, n),
    })
 
    # Inject missing values
    for col in ["budget_musd", "revenue_musd", "rating", "runtime_min"]:
        idx = np.random.choice(df.index, size=int(n * 0.06), replace=False)
        df.loc[idx, col] = np.nan
 
    # Inject outliers
    out_idx = np.random.choice(df.index, size=8, replace=False)
    df.loc[out_idx, "budget_musd"] *= np.random.uniform(15, 25, 8)
    out_idx2 = np.random.choice(df.index, size=5, replace=False)
    df.loc[out_idx2, "rating"] = np.random.choice([0.1, 15, 22], 5)
 
    # Inject duplicate rows
    df = pd.concat([df, df.sample(15, random_state=1)], ignore_index=True)
 
    return df
 
 
# --------------------------------------------------------------------------
# 2. Cleaning functions
# --------------------------------------------------------------------------
def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate rows")
    return df
 
 
def standardize_text(df, column):
    df[column] = df[column].str.strip().str.title()
    return df
 
 
def handle_missing_values(df):
    """Fills missing values using sensible strategies per column."""
    df["rating"] = df["rating"].fillna(df["rating"].median())
    df["runtime_min"] = df["runtime_min"].fillna(df["runtime_min"].median())
 
    # Budget/revenue filled with the median WITHIN each genre (more accurate
    # than a single global median, since typical budgets vary a lot by genre)
    df["budget_musd"] = df["budget_musd"].fillna(
        df.groupby("genre")["budget_musd"].transform("median")
    )
    df["revenue_musd"] = df["revenue_musd"].fillna(
        df.groupby("genre")["revenue_musd"].transform("median")
    )
    return df
 
 
def handle_outliers(df):
    """
    Removes impossible values (e.g. ratings outside 1-10) and caps
    statistical outliers in budget using the IQR method.
    """
    before = len(df)
    df = df[(df["rating"] >= 1) & (df["rating"] <= 10)]
    print(f"Removed {before - len(df)} rows with impossible ratings")
 
    Q1 = df["budget_musd"].quantile(0.25)
    Q3 = df["budget_musd"].quantile(0.75)
    iqr = Q3 - Q1
    upper_bound = Q3 + 1.5 * iqr
 
    n_outliers = (df["budget_musd"] > upper_bound).sum()
    print(f"Capping {n_outliers} budget outliers above ${upper_bound:.1f}M")
    df["budget_musd"] = np.clip(df["budget_musd"], None, upper_bound)
 
    return df
 
 
def engineer_features(df):
    df["profit_musd"] = df["revenue_musd"] - df["budget_musd"]
    df["decade"] = (df["release_year"] // 10 * 10).astype(str) + "s"
    return df
 
 
def clean_dataset(df):
    """Runs the full cleaning pipeline in order."""
    df = remove_duplicates(df)
    df = standardize_text(df, "genre")
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = engineer_features(df)
    return df
 
 
# --------------------------------------------------------------------------
# 3. Visualization / dashboard
# --------------------------------------------------------------------------
def plot_distributions(df, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
    sns.histplot(df["rating"], bins=20, kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title("Distribution of Ratings")
 
    sns.boxplot(y=df["budget_musd"], ax=axes[1], color="salmon")
    axes[1].set_title("Budget Spread (after outlier capping)")
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
def plot_relationships(df, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
    sns.scatterplot(
        data=df, x="budget_musd", y="revenue_musd",
        hue="genre", ax=axes[0], legend=False, alpha=0.6
    )
    axes[0].set_title("Budget vs Revenue")
 
    corr = df[["budget_musd", "revenue_musd", "rating", "runtime_min"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=axes[1])
    axes[1].set_title("Correlation Heatmap")
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
def plot_genre_analysis(df, save_path=None):
    genre_rating = df.groupby("genre")["rating"].mean().sort_values(ascending=False)
    genre_profit = df.groupby("genre")["profit_musd"].mean().sort_values(ascending=False)
 
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
    genre_rating.plot(kind="bar", ax=axes[0], color="seagreen")
    axes[0].set_title("Average Rating by Genre")
    axes[0].tick_params(axis="x", rotation=45)
 
    genre_profit.plot(kind="bar", ax=axes[1], color="goldenrod")
    axes[1].set_title("Average Profit by Genre ($M)")
    axes[1].tick_params(axis="x", rotation=45)
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
    return genre_rating, genre_profit
 
 
def build_dashboard(df, save_path="dashboard.png"):
    """Combines the key charts into a single summary dashboard image."""
    genre_rating = df.groupby("genre")["rating"].mean().sort_values(ascending=False)
 
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
 
    sns.histplot(df["rating"], bins=20, kde=True, ax=axes[0, 0], color="steelblue")
    axes[0, 0].set_title("Rating Distribution")
 
    genre_rating.plot(kind="bar", ax=axes[0, 1], color="seagreen")
    axes[0, 1].set_title("Avg Rating by Genre")
    axes[0, 1].tick_params(axis="x", rotation=45)
 
    sns.scatterplot(
        data=df, x="budget_musd", y="revenue_musd",
        hue="genre", ax=axes[1, 0], legend=False, alpha=0.6
    )
    axes[1, 0].set_title("Budget vs Revenue")
 
    df.groupby("decade")["profit_musd"].mean().plot(kind="bar", ax=axes[1, 1], color="coral")
    axes[1, 1].set_title("Avg Profit by Decade")
    axes[1, 1].tick_params(axis="x", rotation=45)
 
    fig.suptitle("Movie Dataset — Key Insights Dashboard", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Dashboard saved to {save_path}")
 
 
# --------------------------------------------------------------------------
# 4. Main
# --------------------------------------------------------------------------
def main():
    print("Generating raw dataset...")
    raw_df = generate_raw_data()
    raw_df.to_csv("raw_movies.csv", index=False)
    print(f"Raw shape: {raw_df.shape}, duplicates: {raw_df.duplicated().sum()}, "
          f"missing values:\n{raw_df.isnull().sum()}\n")
 
    print("Cleaning dataset...")
    clean_df = clean_dataset(raw_df.copy())
    clean_df.to_csv("cleaned_movies.csv", index=False)
    print(f"\nClean shape: {clean_df.shape}\n")
 
    print("Generating visualizations...")
    plot_distributions(clean_df, save_path="distributions.png")
    plot_relationships(clean_df, save_path="relationships.png")
    genre_rating, genre_profit = plot_genre_analysis(clean_df, save_path="genre_analysis.png")
    build_dashboard(clean_df, save_path="dashboard.png")
 
    print("\nTop-rated genre:", genre_rating.idxmax(), f"({genre_rating.max():.2f})")
    print("Most profitable genre:", genre_profit.idxmax(), f"(${genre_profit.max():.1f}M avg profit)")
 
 
if __name__ == "__main__":
    main()
