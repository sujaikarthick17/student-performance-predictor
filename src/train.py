import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import METADATA_PATH, MODEL_PATH, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from src.data import load_data, validate_data


def main() -> None:
    df = load_data().drop_duplicates().copy()
    feature_columns = validate_data(df)
    df = df.dropna(subset=[TARGET_COLUMN])

    X = df[feature_columns]
    y = df[TARGET_COLUMN].astype(str)
    if y.nunique() < 2:
        raise ValueError("The target column must contain at least two different classes.")

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [c for c in feature_columns if c not in numeric_features]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    metadata = {
        "target_column": TARGET_COLUMN,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "categorical_options": {
            col: sorted(df[col].dropna().astype(str).unique().tolist())
            for col in categorical_features
        },
        "numeric_defaults": {
            col: float(df[col].median()) for col in numeric_features
        },
        "classes": sorted(y.unique().tolist()),
        "test_accuracy": round(float(accuracy), 4),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()
