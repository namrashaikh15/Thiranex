"""
Exploratory Data Analysis: Employee Attrition
------------------------------------------------
Analyzes an HR dataset to uncover patterns and trends behind employee
attrition, using statistical summaries and visualizations, then prints
a structured findings report identifying the key influencing factors.

Author: Namra
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


# --------------------------------------------------------------------------
# 1. Load / generate the dataset
# --------------------------------------------------------------------------
def generate_data(n=1200, seed=42):
    """
    Generates a synthetic HR dataset where attrition realistically depends
    on overtime, satisfaction, work-life balance, tenure, income, distance
    from home, and promotions. Replace this function with
    pd.read_csv("your_file.csv") to use a real dataset instead.
    """
    np.random.seed(seed)

    department = np.random.choice(
        ["Sales", "Engineering", "HR", "Marketing", "Finance", "Operations"],
        n, p=[.22, .28, .08, .12, .15, .15]
    )
    age = np.random.randint(21, 60, n)
    tenure_years = np.round(np.clip(np.random.exponential(4, n), 0, 30), 1)
    monthly_income = np.round(np.random.gamma(6, 700, n) + tenure_years * 150, -1)
    job_satisfaction = np.random.randint(1, 5, n)
    work_life_balance = np.random.randint(1, 5, n)
    overtime = np.random.choice(["Yes", "No"], n, p=[.3, .7])
    distance_from_home_km = np.round(np.clip(np.random.exponential(8, n), 1, 40), 1)
    num_promotions = np.random.poisson(0.7, n)
    performance_rating = np.random.randint(1, 5, n)

    logit = (
        0.0
        + 0.7 * (overtime == "Yes")
        - 0.35 * job_satisfaction
        - 0.25 * work_life_balance
        - 0.15 * tenure_years
        - 0.00003 * monthly_income
        + 0.04 * distance_from_home_km
        - 0.3 * num_promotions
    )
    prob_attrition = 1 / (1 + np.exp(-logit))
    attrition = np.random.binomial(1, prob_attrition)

    df = pd.DataFrame({
        "employee_id": range(1, n + 1),
        "department": department,
        "age": age,
        "tenure_years": tenure_years,
        "monthly_income": monthly_income,
        "job_satisfaction": job_satisfaction,
        "work_life_balance": work_life_balance,
        "overtime": overtime,
        "distance_from_home_km": distance_from_home_km,
        "num_promotions": num_promotions,
        "performance_rating": performance_rating,
        "attrition": np.where(attrition == 1, "Yes", "No"),
    })

    # realistic imperfections
    for col in ["monthly_income", "job_satisfaction", "work_life_balance"]:
        idx = np.random.choice(df.index, size=int(n * 0.04), replace=False)
        df.loc[idx, col] = np.nan
    df = pd.concat([df, df.sample(10, random_state=1)], ignore_index=True)

    return df


# --------------------------------------------------------------------------
# 2. Cleaning
# --------------------------------------------------------------------------
def clean_data(df):
    print("Duplicates:", df.duplicated().sum())
    print("Missing values:\n", df.isnull().sum())

    df = df.drop_duplicates().reset_index(drop=True)
    df["job_satisfaction"] = df["job_satisfaction"].fillna(df["job_satisfaction"].median())
    df["work_life_balance"] = df["work_life_balance"].fillna(df["work_life_balance"].median())
    df["monthly_income"] = df["monthly_income"].fillna(
        df.groupby("department")["monthly_income"].transform("median")
    )
    df["attrition_flag"] = (df["attrition"] == "Yes").astype(int)

    return df


# --------------------------------------------------------------------------
# 3. Statistical summaries
# --------------------------------------------------------------------------
def summarize(df):
    print("\nShape:", df.shape)
    print("\nAttrition rate:\n", df["attrition"].value_counts(normalize=True).round(3))
    print("\nNumeric summary:\n", df.describe().round(2))


def correlation_summary(df):
    num_cols = [
        "age", "tenure_years", "monthly_income", "job_satisfaction",
        "work_life_balance", "distance_from_home_km", "num_promotions",
        "performance_rating", "attrition_flag"
    ]
    corr = df[num_cols].corr()
    print("\nCorrelation with attrition, sorted:\n", corr["attrition_flag"].sort_values().round(3))
    return corr


# --------------------------------------------------------------------------
# 4. Visualizations
# --------------------------------------------------------------------------
def plot_univariate(df, save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    sns.histplot(df["age"], bins=20, ax=axes[0, 0], color="steelblue")
    axes[0, 0].set_title("Age Distribution")

    sns.histplot(df["tenure_years"], bins=20, ax=axes[0, 1], color="salmon")
    axes[0, 1].set_title("Tenure (Years)")

    sns.histplot(df["monthly_income"], bins=25, ax=axes[0, 2], color="seagreen")
    axes[0, 2].set_title("Monthly Income")

    df["department"].value_counts().plot(kind="bar", ax=axes[1, 0], color="goldenrod")
    axes[1, 0].set_title("Employees by Department")
    axes[1, 0].tick_params(axis="x", rotation=45)

    df["job_satisfaction"].value_counts().sort_index().plot(kind="bar", ax=axes[1, 1], color="mediumpurple")
    axes[1, 1].set_title("Job Satisfaction (1=Low, 4=High)")

    df["overtime"].value_counts().plot(kind="bar", ax=axes[1, 2], color="indianred")
    axes[1, 2].set_title("Overtime")
    axes[1, 2].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_attrition_factors(df, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    overtime_rate = df.groupby("overtime")["attrition_flag"].mean()
    overtime_rate.plot(kind="bar", ax=axes[0], color=["seagreen", "indianred"])
    axes[0].set_title("Attrition Rate by Overtime")
    axes[0].tick_params(axis="x", rotation=0)

    dept_rate = df.groupby("department")["attrition_flag"].mean().sort_values(ascending=False)
    dept_rate.plot(kind="bar", ax=axes[1], color="coral")
    axes[1].set_title("Attrition Rate by Department")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return overtime_rate, dept_rate


def plot_boxplots(df, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sns.boxplot(data=df, x="attrition", y="job_satisfaction", hue="attrition", ax=axes[0], palette="Set2", legend=False)
    axes[0].set_title("Job Satisfaction by Attrition")

    sns.boxplot(data=df, x="attrition", y="tenure_years", hue="attrition", ax=axes[1], palette="Set2", legend=False)
    axes[1].set_title("Tenure by Attrition")

    sns.boxplot(data=df, x="attrition", y="monthly_income", hue="attrition", ax=axes[2], palette="Set2", legend=False)
    axes[2].set_title("Monthly Income by Attrition")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_correlation_heatmap(corr, save_path=None):
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_multifactor(df, save_path=None):
    pivot = df.pivot_table(
        index="department", columns="overtime", values="attrition_flag", aggfunc="mean"
    ).round(3)

    pivot.plot(kind="bar", figsize=(10, 5), color=["seagreen", "indianred"])
    plt.title("Attrition Rate by Department, Split by Overtime")
    plt.ylabel("Attrition Rate")
    plt.xticks(rotation=45)
    plt.legend(title="Overtime")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return pivot


# --------------------------------------------------------------------------
# 5. Structured findings report
# --------------------------------------------------------------------------
def print_report(df, corr, overtime_rate, dept_rate):
    print("\n" + "=" * 60)
    print("STRUCTURED FINDINGS REPORT")
    print("=" * 60)

    print(f"\nDataset: {df.shape[0]} employees, overall attrition rate "
          f"{df['attrition_flag'].mean():.1%}")

    print(f"\nOvertime effect: employees working overtime leave at "
          f"{overtime_rate['Yes']:.1%} vs {overtime_rate['No']:.1%} for those who don't.")

    top_dept = dept_rate.idxmax()
    print(f"\nHighest attrition department: {top_dept} ({dept_rate.max():.1%})")

    print("\nTop correlates with attrition (numeric factors):")
    print(corr["attrition_flag"].drop("attrition_flag").abs().sort_values(ascending=False).round(3))

    leavers = df[df["attrition"] == "Yes"]
    stayers = df[df["attrition"] == "No"]
    print(f"\nAvg tenure — leavers: {leavers['tenure_years'].mean():.1f} yrs, "
          f"stayers: {stayers['tenure_years'].mean():.1f} yrs")
    print(f"Avg satisfaction — leavers: {leavers['job_satisfaction'].mean():.2f}, "
          f"stayers: {stayers['job_satisfaction'].mean():.2f}")

    print("\nRecommendations:")
    print("1. Review overtime policy in high-attrition departments — strongest single lever found.")
    print("2. Focus retention efforts on employees in their first 1-3 years.")
    print("3. Track satisfaction/work-life balance as leading indicators, not just exit surveys.")
    print("=" * 60)


# --------------------------------------------------------------------------
# 6. Main
# --------------------------------------------------------------------------
def main():
    print("Generating dataset...")
    raw_df = generate_data()
    raw_df.to_csv("hr_data_raw.csv", index=False)

    df = clean_data(raw_df.copy())
    df.to_csv("hr_data_cleaned.csv", index=False)

    summarize(df)
    corr = correlation_summary(df)

    plot_univariate(df, save_path="univariate.png")
    overtime_rate, dept_rate = plot_attrition_factors(df, save_path="attrition_factors.png")
    plot_boxplots(df, save_path="boxplots.png")
    plot_correlation_heatmap(corr, save_path="correlation_heatmap.png")
    plot_multifactor(df, save_path="multifactor.png")

    print_report(df, corr, overtime_rate, dept_rate)


if __name__ == "__main__":
    main()
