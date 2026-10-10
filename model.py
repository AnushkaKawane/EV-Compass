from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    ROOT
    / "data"
    / "EV_Adoption_and_Range_Anxiety_Dataset.csv"
)

MODEL_DIR = ROOT / "models"

MODEL_PATH = MODEL_DIR / "ev_model.joblib"

SUMMARY_PATH = MODEL_DIR / "dataset_summary.json"

METRICS_PATH = MODEL_DIR / "model_metrics.json"


# --------------------------------------------------
# DATASET CONFIGURATION
# --------------------------------------------------

TARGET = "Will_Buy_EV"

NUMERIC_FEATURES = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------
# LOAD AND VALIDATE DATASET
# --------------------------------------------------

def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = set(FEATURES + [TARGET])

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: "
            f"{sorted(missing_columns)}"
        )

    df = df.drop_duplicates().copy()

    df[TARGET] = (
        df[TARGET]
        .astype(str)
        .str.strip()
        .str.title()
    )

    df = df[
        df[TARGET].isin(["Yes", "No"])
    ].copy()

    if df.empty:
        raise ValueError(
            "No valid Yes/No labels found in Will_Buy_EV."
        )

    return df


# --------------------------------------------------
# BUILD MACHINE LEARNING PIPELINE
# --------------------------------------------------

def build_pipeline():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )

    return pipeline


# --------------------------------------------------
# CREATE DATASET SUMMARY
# --------------------------------------------------

def create_dataset_summary(df):

    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "feature_count": len(FEATURES),
        "target": TARGET,

        "target_counts": {
            str(key): int(value)
            for key, value
            in df[TARGET].value_counts().items()
        },

        "missing_values": {
            str(key): int(value)
            for key, value
            in df.isna().sum().items()
        },

        "numeric_means": {
            column: (
                None
                if pd.isna(df[column].mean())
                else round(
                    float(df[column].mean()),
                    2,
                )
            )
            for column in NUMERIC_FEATURES
        },

        "range_anxiety_counts": {
            str(key): int(value)
            for key, value
            in df[
                "Range_Anxiety_Level"
            ].value_counts().items()
        },

        "city_type_counts": {
            str(key): int(value)
            for key, value
            in df["City_Type"].value_counts().items()
        },

        "buy_rate_percent": round(
            float(
                (df[TARGET] == "Yes").mean()
                * 100
            ),
            2,
        ),
    }

    return summary


# --------------------------------------------------
# TRAIN AND EVALUATE MODEL
# --------------------------------------------------

def train_model():

    df = load_dataset()

    X = df[FEATURES].copy()

    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    pipeline = build_pipeline()

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    metrics = {
        "test_rows": int(len(y_test)),

        "accuracy": round(
            float(
                accuracy_score(
                    y_test,
                    predictions,
                )
            ),
            4,
        ),

        "classification_report": (
            classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0,
            )
        ),
    }

    if hasattr(pipeline, "predict_proba"):

        positive_index = list(
            pipeline.classes_
        ).index("Yes")

        probabilities = pipeline.predict_proba(
            X_test
        )[:, positive_index]

        metrics["roc_auc"] = round(
            float(
                roc_auc_score(
                    y_test == "Yes",
                    probabilities,
                )
            ),
            4,
        )

    else:
        metrics["roc_auc"] = None

    # Fit the deployment model using all valid records.
    pipeline.fit(X, y)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "pipeline": pipeline,
            "features": FEATURES,
            "target": TARGET,
        },
        MODEL_PATH,
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            create_dataset_summary(df),
            indent=2,
        ),
        encoding="utf-8",
    )

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Model training completed.")

    print(
        f"Dataset records: {len(df)}"
    )

    print(
        f"Hold-out accuracy: "
        f"{metrics['accuracy']:.3f}"
    )

    if metrics["roc_auc"] is not None:
        print(
            f"Hold-out ROC-AUC: "
            f"{metrics['roc_auc']:.3f}"
        )

    print(
        f"Saved model: {MODEL_PATH}"
    )

    return MODEL_PATH


# --------------------------------------------------
# EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    train_model()