from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR / "EV_Adoption_and_Range_Anxiety_Dataset.csv"
)

MODEL_DIR = BASE_DIR / "saved_model"

MODEL_PATH = MODEL_DIR / "ev_model.joblib"

METRICS_PATH = MODEL_DIR / "metrics.json"


# --------------------------------------------------
# DATASET CONFIGURATION
# --------------------------------------------------

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

TARGET = "Will_Buy_EV"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing these columns: "
            + ", ".join(missing_columns)
        )

    df = df[required_columns].copy()

    # Remove rows without a valid target.
    df = df.dropna(subset=[TARGET])

    # Normalize target values.
    df[TARGET] = (
        df[TARGET]
        .astype(str)
        .str.strip()
        .str.title()
    )

    df = df[df[TARGET].isin(["Yes", "No"])]

    if df[TARGET].nunique() < 2:
        raise ValueError(
            "The target must contain both Yes and No."
        )

    # Convert numerical columns to numeric values.
    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Normalize categorical columns.
    for column in CATEGORICAL_FEATURES:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

        df[column] = df[column].replace(
            "",
            pd.NA
        )

    return df


# --------------------------------------------------
# CREATE THE MACHINE LEARNING PIPELINE
# --------------------------------------------------

def create_pipeline():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            ),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )

    return pipeline


# --------------------------------------------------
# TRAIN AND EVALUATE MODEL
# --------------------------------------------------

def train_model():

    print("Loading dataset...")

    df = load_dataset()

    X = df[FEATURES]

    y = df[TARGET].map({
        "No": 0,
        "Yes": 1
    })

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    print("Training Random Forest model...")

    pipeline = create_pipeline()

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    auc = None

    if y_test.nunique() == 2:
        auc = roc_auc_score(
            y_test,
            probabilities
        )

    metrics = {
        "accuracy": round(
            float(accuracy),
            4
        ),
        "roc_auc": (
            round(float(auc), 4)
            if auc is not None
            else None
        ),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "dataset_rows": int(len(df)),
        "target_distribution": {
            str(key): int(value)
            for key, value in (
                df[TARGET]
                .value_counts()
                .to_dict()
                .items()
            )
        }
    }

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4
        )

    print("\nModel training completed.")

    print(
        f"Accuracy: {accuracy:.2%}"
    )

    if auc is not None:
        print(
            f"ROC-AUC: {auc:.4f}"
        )

    print(
        f"Saved model: {MODEL_PATH}"
    )

    print(
        f"Saved metrics: {METRICS_PATH}"
    )

    return pipeline, metrics


# --------------------------------------------------
# LOAD EXISTING MODEL OR TRAIN A NEW ONE
# --------------------------------------------------

def get_model():

    if MODEL_PATH.exists():
        try:
            pipeline = joblib.load(
                MODEL_PATH
            )

            if METRICS_PATH.exists():
                with open(
                    METRICS_PATH,
                    "r",
                    encoding="utf-8"
                ) as file:
                    metrics = json.load(file)
            else:
                metrics = {}

            return pipeline, metrics

        except Exception as error:
            print(
                "Saved model could not be loaded:",
                error
            )

    return train_model()


# --------------------------------------------------
# PREDICT PURCHASE LIKELIHOOD
# --------------------------------------------------

def predict_probability(
    pipeline,
    user_data
):

    input_df = pd.DataFrame(
        [user_data],
        columns=FEATURES
    )

    probabilities = pipeline.predict_proba(
        input_df
    )[0]

    classes = list(
        pipeline.named_steps[
            "classifier"
        ].classes_
    )

    yes_index = classes.index(1)

    return float(
        probabilities[yes_index]
    )


# --------------------------------------------------
# RUN THIS FILE DIRECTLY
# --------------------------------------------------

if __name__ == "__main__":
    train_model()