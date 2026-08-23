import matplotlib.pyplot as plt
import seaborn as sns

from src.config import REPORTS_DIR, TARGET_COLUMN
from src.data import load_data, validate_data


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    df = load_data()
    features = validate_data(df)

    print("\n========== DATASET PREVIEW ==========")
    print(df.head())
    print("\n========== SIZE (rows, columns) ==========")
    print(df.shape)
    print("\n========== COLUMNS ==========")
    print(df.columns.tolist())
    print("\n========== DATA TYPES / NON-NULL COUNTS ==========")
    df.info()
    print("\n========== STATISTICAL SUMMARY ==========")
    print(df.describe(include="all").transpose())
    print("\n========== MISSING VALUES ==========")
    print(df.isna().sum())
    print("\n========== DUPLICATE ROWS ==========")
    print(df.duplicated().sum())
    print("\n========== TARGET COUNTS ==========")
    print(df[TARGET_COLUMN].value_counts(dropna=False))

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x=TARGET_COLUMN, order=df[TARGET_COLUMN].value_counts().index)
    plt.title("Student performance distribution")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "target_distribution.png", dpi=150)
    plt.close()

    numeric_features = df[features].select_dtypes(include="number").columns.tolist()
    if numeric_features:
        df[numeric_features].hist(figsize=(12, 8), bins=15)
        plt.suptitle("Numeric feature distributions")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "numeric_distributions.png", dpi=150)
        plt.close()

    if "study_hours" in df.columns and "internal_marks" in df.columns:
        plt.figure(figsize=(7, 4))
        sns.scatterplot(data=df, x="study_hours", y="internal_marks", hue=TARGET_COLUMN)
        plt.title("Study hours vs internal marks")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "study_hours_vs_marks.png", dpi=150)
        plt.close()

    print(f"\nEDA complete. Charts saved in: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
