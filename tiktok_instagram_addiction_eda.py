import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PLATFORM_COLORS = {"TikTok": "#ff0050", "Instagram": "#c13584"}

# ============================================================
#  1. LOAD DATA
# ============================================================

def load_data(filepath: str = "social_media_addiction.csv") -> pd.DataFrame:
    """Load dataset. Replace filepath with your actual file."""
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ============================================================
#  2. DATA OVERVIEW
# ============================================================

def data_overview(df: pd.DataFrame):
    print("\n" + "="*55)
    print("  DATA OVERVIEW")
    print("="*55)
    print(df.dtypes.to_string())
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nDuplicates: {df.duplicated().sum()}")
    print(f"\nNumerical summary:\n{df.describe().round(2).T.to_string()}")


# ============================================================
#  3. DATA CLEANING
# ============================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()

    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    
    if "platform" in df.columns:
        df["platform"] = df["platform"].str.strip().str.title()

   
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 17, 24, 34, 44, 100],
            labels=["<18", "18–24", "25–34", "35–44", "45+"]
        )

    print("✓ Data cleaned")
    return df


# ============================================================
#  4. UNIVARIATE ANALYSIS
# ============================================================

def univariate_analysis(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Univariate Analysis — Key Variables", fontsize=14, y=1.01)

    num_vars = [
        "daily_usage_hours", "sessions_per_day",
        "avg_session_minutes", "addiction_score",
        "sleep_hours", "anxiety_score"
    ]

    for ax, col in zip(axes.flat, num_vars):
        if col in df.columns:
            sns.histplot(df[col], kde=True, ax=ax, color="#5c6bc0")
            ax.set_title(col.replace("_", " ").title())
            ax.set_xlabel("")
        else:
            ax.set_visible(False)

    plt.tight_layout()
    plt.savefig("01_univariate.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Saved: 01_univariate.png")


def platform_distribution(df: pd.DataFrame):
    if "platform" not in df.columns:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Platform Distribution", fontsize=13)

    counts = df["platform"].value_counts()
    counts.plot.pie(
        ax=ax1, autopct="%1.1f%%",
        colors=[PLATFORM_COLORS.get(p, "#888") for p in counts.index],
        startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    ax1.set_ylabel("")
    ax1.set_title("Share of users")

    sns.countplot(data=df, x="platform", palette=PLATFORM_COLORS, ax=ax2)
    ax2.set_title("Count by platform")
    ax2.set_xlabel("")

    plt.tight_layout()
    plt.savefig("02_platform_dist.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Saved: 02_platform_dist.png")


# ============================================================
#  5. BIVARIATE ANALYSIS
# ============================================================

def usage_vs_addiction(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Usage vs Addiction Score", fontsize=13)

    for ax, xcol in zip(axes, ["daily_usage_hours", "sessions_per_day"]):
        if xcol not in df.columns or "addiction_score" not in df.columns:
            continue
        hue = "platform" if "platform" in df.columns else None
        palette = PLATFORM_COLORS if hue else None
        sns.scatterplot(
            data=df, x=xcol, y="addiction_score",
            hue=hue, palette=palette, alpha=0.55, ax=ax
        )
        
        m, b, r, p, _ = stats.linregress(df[xcol], df["addiction_score"])
        xs = np.linspace(df[xcol].min(), df[xcol].max(), 100)
        ax.plot(xs, m*xs + b, color="black", linewidth=1.5, linestyle="--",
                label=f"r={r:.2f}, p={p:.3f}")
        ax.legend(fontsize=9)
        ax.set_title(xcol.replace("_", " ").title())

    plt.tight_layout()
    plt.savefig("03_usage_vs_addiction.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Saved: 03_usage_vs_addiction.png")


def platform_comparison(df: pd.DataFrame):
    metrics = [
        "daily_usage_hours", "addiction_score",
        "anxiety_score", "sleep_hours"
    ]
    available = [m for m in metrics if m in df.columns]
    if "platform" not in df.columns or not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    fig.suptitle("TikTok vs Instagram — Key Metrics", fontsize=13)

    for ax, col in zip(axes, available):
        sns.boxplot(
            data=df, x="platform", y=col,
            palette=PLATFORM_COLORS, ax=ax, width=0.5
        )
       
        groups = [g[col].dropna().values for _, g in df.groupby("platform")]
        if len(groups) == 2:
            _, p = stats.mannwhitneyu(*groups, alternative="two-sided")
            ax.set_title(f"{col.replace('_',' ').title()}\np={p:.3f}")
        else:
            ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("")

    plt.tight_layout()
    plt.savefig("04_platform_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Saved: 04_platform_comparison.png")


# ============================================================
#  6. DEMOGRAPHIC BREAKDOWN
# ============================================================

def demographics(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Addiction Score by Demographics", fontsize=13)

    if "age_group" in df.columns and "addiction_score" in df.columns:
        hue = "platform" if "platform" in df.columns else None
        sns.barplot(
            data=df, x="age_group", y="addiction_score",
            hue=hue, palette=PLATFORM_COLORS if hue else "muted",
            ax=axes[0], ci=95, capsize=0.1
        )
        axes[0].set_title("By age group")
        axes[0].set_xlabel("Age group")

    if "gender" in df.columns and "addiction_score" in df.columns:
        sns.violinplot(
            data=df, x="gender", y="addiction_score",
            palette="pastel", ax=axes[1], inner="box"
        )
        axes[1].set_title("By gender")
        axes[1].set_xlabel("")

    plt.tight_layout()
    plt.savefig("05_demographics.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Saved: 05_demographics.png")


# ============================================================
#  7. CORRELATION HEATMAP
# ============================================================

def correlation_heatmap(df: pd.DataFrame):
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 3:
        return

    fig, ax = plt.subplots(figsize=(11, 8))
    mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
    sns.heatmap(
        num_df.corr(), mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.4,
        ax=ax, annot_kws={"size": 9}
    )
    ax.set_title("Correlation Matrix — All Numerical Variables", fontsize=13)
    plt.tight_layout()
    plt.savefig("06_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Saved: 06_correlation_heatmap.png")


# ============================================================
#  8. SLEEP & MENTAL HEALTH IMPACT
# ============================================================

def mental_health_impact(df: pd.DataFrame):
    cols = ["addiction_score", "anxiety_score", "depression_score",
            "sleep_hours", "productivity_score"]
    available = [c for c in cols if c in df.columns]
    if len(available) < 2:
        return

    g = sns.PairGrid(
        df[available + (["platform"] if "platform" in df.columns else [])],
        hue="platform" if "platform" in df.columns else None,
        palette=PLATFORM_COLORS,
        diag_sharey=False
    )
    g.map_upper(sns.scatterplot, alpha=0.4, s=20)
    g.map_lower(sns.kdeplot, fill=True, alpha=0.3)
    g.map_diag(sns.histplot, kde=True)
    g.add_legend()
    g.figure.suptitle("Mental Health & Sleep — Pairplot", y=1.01, fontsize=13)
    plt.savefig("07_mental_health_pairplot.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("✓ Saved: 07_mental_health_pairplot.png")


# ============================================================
#  9. ADDICTION RISK SEGMENTATION
# ============================================================

def risk_segmentation(df: pd.DataFrame):
    if "addiction_score" not in df.columns:
        return

    df = df.copy()
    score_min, score_max = df["addiction_score"].min(), df["addiction_score"].max()
    bins = np.linspace(score_min, score_max, 4)
    df["risk_level"] = pd.cut(
        df["addiction_score"],
        bins=bins,
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Addiction Risk Segmentation", fontsize=13)

    risk_palette = {"Low": "#43a047", "Medium": "#fb8c00", "High": "#e53935"}
    sns.countplot(data=df, x="risk_level", order=["Low","Medium","High"],
                  palette=risk_palette, ax=axes[0])
    axes[0].set_title("User count by risk level")
    axes[0].set_xlabel("")

    if "platform" in df.columns:
        cross = pd.crosstab(df["platform"], df["risk_level"], normalize="index") * 100
        cross[["Low","Medium","High"]].plot(
            kind="bar", stacked=True, ax=axes[1],
            color=[risk_palette[l] for l in ["Low","Medium","High"]],
            edgecolor="white"
        )
        axes[1].set_title("Risk distribution per platform (%)")
        axes[1].set_xlabel("")
        axes[1].legend(title="Risk", bbox_to_anchor=(1.01, 1))
        axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig("08_risk_segmentation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Saved: 08_risk_segmentation.png")


# ============================================================
#  10. SUMMARY STATS TABLE
# ============================================================

def summary_table(df: pd.DataFrame):
    if "platform" not in df.columns:
        return
    metrics = [
        "daily_usage_hours", "sessions_per_day",
        "addiction_score", "anxiety_score", "sleep_hours"
    ]
    available = [m for m in metrics if m in df.columns]
    table = df.groupby("platform")[available].agg(["mean","median","std"]).round(2)
    print("\n" + "="*55)
    print("  SUMMARY STATISTICS BY PLATFORM")
    print("="*55)
    print(table.to_string())
    table.to_csv("09_summary_stats.csv")
    print("✓ Saved: 09_summary_stats.csv")




if __name__ == "__main__":

    FILE = "social_media_addiction.csv"

    df = load_data(FILE)
    data_overview(df)
    df = clean_data(df)

    
    univariate_analysis(df)
    platform_distribution(df)
    usage_vs_addiction(df)
    platform_comparison(df)
    demographics(df)
    correlation_heatmap(df)
    mental_health_impact(df)
    risk_segmentation(df)
    summary_table(df)

    print("\n✅ EDA complete. All charts saved as PNG files.")
