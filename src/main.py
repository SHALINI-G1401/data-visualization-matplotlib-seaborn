
"""
PROJECT 02: Data Visualization with Matplotlib & Seaborn
Dataset: Seaborn Tips Dataset
Author: Your Name
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "tips.csv")
CHART_DIR = os.path.join(BASE_DIR, "charts")

os.makedirs(CHART_DIR, exist_ok=True)

sns.set_theme(
    style="whitegrid",
    palette="colorblind",
    font_scale=1.0
)

print("=" * 60)
print("PROJECT 02: DATA VISUALIZATION")
print("=" * 60)


# --------------------------------------------------
# 2. LOAD AND VALIDATE DATASET
# --------------------------------------------------

def load_data():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "total_bill", "tip", "sex",
        "smoker", "day", "time", "size"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    print("\nDATASET LOADED SUCCESSFULLY")
    print("-" * 40)

    print("Shape:", df.shape)
    print("\nFirst five records:")
    print(df.head())

    return df


# --------------------------------------------------
# 3. DATA EXPLORATION
# --------------------------------------------------

def explore_data(df):

    print("\nDATASET INFORMATION")
    print("-" * 40)

    df.info()

    print("\nSTATISTICAL SUMMARY")
    print("-" * 40)

    print(df.describe())

    print("\nMISSING VALUES")
    print("-" * 40)

    print(df.isnull().sum())

    # Duplicate record count
    print("\nDUPLICATE RECORDS")
    print("-" * 40)

    print(df.duplicated().sum())

    # NEW: Display duplicate row details
    print("\nDUPLICATE ROW DETAILS")
    print("-" * 40)

    duplicate_rows = df[df.duplicated(keep=False)]

    if duplicate_rows.empty:
        print("No duplicate rows found.")
    else:
        print(duplicate_rows)

    print("\nNUMERICAL CORRELATION")
    print("-" * 40)

    print(
        df.select_dtypes(include="number").corr()
    )


# --------------------------------------------------
# 4. PREPARE DATA AND TAKEAWAYS
# --------------------------------------------------

def generate_takeaways(df):

    numeric_df = df.select_dtypes(include="number")

    correlation = numeric_df.corr()

    # Find strongest numeric relationship
    corr_pairs = correlation.where(
        np.triu(
            np.ones(correlation.shape),
            k=1
        ).astype(bool)
    ).stack()

    strongest_pair = corr_pairs.abs().idxmax()

    strongest_value = correlation.loc[
        strongest_pair[0],
        strongest_pair[1]
    ]

    # Day-wise tip statistics
    day_medians = df.groupby("day")["tip"].median()

    highest_median_day = day_medians.idxmax()
    lowest_median_day = day_medians.idxmin()

    # Average tip by day
    average_tips = df.groupby("day")["tip"].mean()

    highest_average_day = average_tips.idxmax()

    # Bill distribution
    median_bill = df["total_bill"].median()
    upper_bill = df["total_bill"].quantile(0.75)

    # Scatter relationship
    bill_tip_corr = df["total_bill"].corr(df["tip"])

    if bill_tip_corr > 0:
        scatter_text = (
            f"Total bill and tip have a positive "
            f"correlation of {bill_tip_corr:.2f}."
        )

    elif bill_tip_corr < 0:
        scatter_text = (
            f"Total bill and tip have a negative "
            f"correlation of {bill_tip_corr:.2f}."
        )

    else:
        scatter_text = (
            "Total bill and tip have almost no linear correlation."
        )

    takeaways = {
        "Histogram": (
            f"The median restaurant bill is ${median_bill:.2f}, "
            f"and 75% of bills are at or below ${upper_bill:.2f}."
        ),

        "Box Plot": (
            f"{highest_median_day} has the highest median tip, "
            f"while {lowest_median_day} has the lowest median tip."
        ),

        "Scatter Plot": scatter_text,

        "Correlation Heatmap": (
            f"The strongest numerical relationship is between "
            f"{strongest_pair[0]} and {strongest_pair[1]}, "
            f"with a correlation of {strongest_value:.2f}."
        ),

        "Bar Chart": (
            f"{highest_average_day} has the highest average tip "
            f"at ${average_tips.max():.2f}."
        )
    }

    return takeaways


# --------------------------------------------------
# 5. HISTOGRAM
# --------------------------------------------------

def plot_histogram(df, ax):

    sns.histplot(
        data=df,
        x="total_bill",
        bins=20,
        kde=True,
        color="#3977B8",
        ax=ax
    )

    median_bill = df["total_bill"].median()

    ax.axvline(
        median_bill,
        color="#D95F02",
        linestyle="--",
        label=f"Median: ${median_bill:.2f}"
    )

    ax.set_title(
        "Restaurant Bill Distribution",
        fontweight="bold"
    )

    ax.set_xlabel("Total Bill (USD)")
    ax.set_ylabel("Number of Meals")
    ax.legend()


# --------------------------------------------------
# 6. BOX PLOT
# --------------------------------------------------

def plot_boxplot(df, ax):

    day_order = ["Thur", "Fri", "Sat", "Sun"]

    day_order = [
        day for day in day_order
        if day in df["day"].unique()
    ]

    sns.boxplot(
        data=df,
        x="day",
        y="tip",
        order=day_order,
        palette="colorblind",
        hue="day",
        legend=False,
        ax=ax
    )

    ax.set_title(
        "Tip Distribution Across Different Days",
        fontweight="bold"
    )

    ax.set_xlabel("Day of the Week")
    ax.set_ylabel("Tip Amount (USD)")


# --------------------------------------------------
# 7. SCATTER PLOT
# --------------------------------------------------

def plot_scatterplot(df, ax):

    sns.scatterplot(
        data=df,
        x="total_bill",
        y="tip",
        hue="time",
        palette="colorblind",
        s=75,
        alpha=0.8,
        ax=ax
    )

    ax.set_title(
        "Relationship Between Bill and Tip",
        fontweight="bold"
    )

    ax.set_xlabel("Total Bill (USD)")
    ax.set_ylabel("Tip Amount (USD)")

    ax.legend(title="Meal Time")


# --------------------------------------------------
# 8. CORRELATION HEATMAP
# --------------------------------------------------

def plot_heatmap(df, ax):

    correlation = df.select_dtypes(
        include="number"
    ).corr()

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        linewidths=0.5,
        square=True,
        vmin=-1,
        vmax=1,
        ax=ax
    )

    ax.set_title(
        "Correlation Between Numerical Variables",
        fontweight="bold"
    )


# --------------------------------------------------
# 9. BAR CHART
# --------------------------------------------------

def plot_barchart(df, ax):

    average_tips = df.groupby("day")["tip"].mean()

    colors = sns.color_palette(
        "colorblind",
        len(average_tips)
    )

    bars = ax.bar(
        average_tips.index,
        average_tips.values,
        color=colors
    )

    ax.bar_label(
        bars,
        fmt="%.2f",
        padding=3
    )

    ax.set_title(
        f"{average_tips.idxmax()} Has the Highest Average Tip",
        fontweight="bold"
    )

    ax.set_xlabel("Day of the Week")
    ax.set_ylabel("Average Tip (USD)")


# --------------------------------------------------
# 10. SAVE INDIVIDUAL CHARTS
# --------------------------------------------------

def save_individual_chart(df, plot_function, filename):

    fig, ax = plt.subplots(figsize=(9, 6))

    plot_function(df, ax)

    fig.tight_layout()

    save_path = os.path.join(CHART_DIR, filename)

    fig.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {filename}")


def generate_individual_charts(df):

    print("\nGENERATING INDIVIDUAL CHARTS")
    print("-" * 40)

    save_individual_chart(
        df,
        plot_histogram,
        "histogram.png"
    )

    save_individual_chart(
        df,
        plot_boxplot,
        "boxplot.png"
    )

    save_individual_chart(
        df,
        plot_scatterplot,
        "scatterplot.png"
    )

    save_individual_chart(
        df,
        plot_heatmap,
        "heatmap.png"
    )

    save_individual_chart(
        df,
        plot_barchart,
        "bar_chart.png"
    )


# --------------------------------------------------
# 11. CREATE 2 x 2 DASHBOARD
# --------------------------------------------------

def create_dashboard(df):

    print("\nCREATING 2 x 2 DASHBOARD")
    print("-" * 40)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10)
    )

    plot_histogram(df, axes[0, 0])

    plot_boxplot(df, axes[0, 1])

    plot_scatterplot(df, axes[1, 0])

    plot_heatmap(df, axes[1, 1])

    fig.suptitle(
        "RESTAURANT TIPS - DATA VISUALIZATION DASHBOARD",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    dashboard_path = os.path.join(
        CHART_DIR,
        "tips_overview.png"
    )

    fig.savefig(
        dashboard_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Dashboard saved: tips_overview.png")


# --------------------------------------------------
# 12. SAVE WRITTEN TAKEAWAYS
# --------------------------------------------------

def save_takeaways(takeaways):

    output_path = os.path.join(
        CHART_DIR,
        "takeaways.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "PROJECT 02 - DATA VISUALIZATION TAKEAWAYS\n"
        )

        file.write("=" * 55 + "\n\n")

        for title, takeaway in takeaways.items():

            file.write(title + "\n")

            file.write("-" * 40 + "\n")

            file.write(takeaway + "\n\n")

    print("Takeaways saved: takeaways.txt")


# --------------------------------------------------
# 13. MAIN EXECUTION
# --------------------------------------------------

def main():

    try:

        df = load_data()

        explore_data(df)

        takeaways = generate_takeaways(df)

        generate_individual_charts(df)

        create_dashboard(df)

        save_takeaways(takeaways)

        print("\nPROJECT TAKEAWAYS")
        print("=" * 60)

        for title, takeaway in takeaways.items():

            print(f"\n{title}:")
            print(takeaway)

        print("\n" + "=" * 60)
        print("PROJECT COMPLETED SUCCESSFULLY!")
        print("Check the charts folder for your output.")
        print("=" * 60)

    except Exception as error:

        print("\nPROJECT EXECUTION FAILED")
        print("-" * 40)

        print(type(error).__name__, ":", error)

        sys.exit(1)


if __name__ == "__main__":
    main()
