from pathlib import Path
import json

import joblib
import pandas as pd

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
)

from model import (
    FEATURES,
    MODEL_DIR,
    MODEL_PATH,
    SUMMARY_PATH,
    train_model,
)


ROOT = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

def ensure_model():

    if not MODEL_PATH.exists():
        train_model()

    return joblib.load(MODEL_PATH)


MODEL_BUNDLE = ensure_model()

PIPELINE = MODEL_BUNDLE["pipeline"]


# --------------------------------------------------
# READ JSON FILES
# --------------------------------------------------

def read_json(path):

    if not path.exists():
        return {}

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return {}


# --------------------------------------------------
# VALIDATE USER INPUT
# --------------------------------------------------

def clean_payload(payload):

    numeric = {
        "Age": int(
            payload.get("Age", 35)
        ),

        "Annual_Income_USD": float(
            payload.get(
                "Annual_Income_USD",
                65000,
            )
        ),

        "Daily_Commute_km": float(
            payload.get(
                "Daily_Commute_km",
                25,
            )
        ),

        "Number_of_Cars_Owned": int(
            payload.get(
                "Number_of_Cars_Owned",
                1,
            )
        ),

        "Charging_Stations_Near_Home": int(
            payload.get(
                "Charging_Stations_Near_Home",
                3,
            )
        ),

        "Charging_Stations_Near_Work": int(
            payload.get(
                "Charging_Stations_Near_Work",
                3,
            )
        ),

        "Environmental_Concern_Level": int(
            payload.get(
                "Environmental_Concern_Level",
                3,
            )
        ),
    }

    categorical = {
        "Gender": str(
            payload.get(
                "Gender",
                "Other",
            )
        ),

        "City_Type": str(
            payload.get(
                "City_Type",
                "Suburban",
            )
        ),

        "Current_Car_Type": str(
            payload.get(
                "Current_Car_Type",
                "Sedan",
            )
        ),

        "Home_Charging_Possible": str(
            payload.get(
                "Home_Charging_Possible",
                "No",
            )
        ),

        "Subsidy_Available": str(
            payload.get(
                "Subsidy_Available",
                "No",
            )
        ),

        "Range_Anxiety_Level": str(
            payload.get(
                "Range_Anxiety_Level",
                "Medium",
            )
        ),
    }

    bounds = {
        "Age": (18, 100),

        "Annual_Income_USD": (
            0,
            10_000_000,
        ),

        "Daily_Commute_km": (
            0,
            1000,
        ),

        "Number_of_Cars_Owned": (
            0,
            20,
        ),

        "Charging_Stations_Near_Home": (
            0,
            100,
        ),

        "Charging_Stations_Near_Work": (
            0,
            100,
        ),

        "Environmental_Concern_Level": (
            1,
            5,
        ),
    }

    for key, (low, high) in bounds.items():

        if not low <= numeric[key] <= high:

            raise ValueError(
                f"{key.replace('_', ' ')} "
                f"must be between {low} and {high}."
            )

    allowed = {
        "Gender": {
            "Male",
            "Female",
            "Other",
        },

        "City_Type": {
            "Urban",
            "Suburban",
            "Rural",
        },

        "Current_Car_Type": {
            "Hatchback",
            "SUV",
            "Sedan",
            "Truck",
        },

        "Home_Charging_Possible": {
            "Yes",
            "No",
        },

        "Subsidy_Available": {
            "Yes",
            "No",
        },

        "Range_Anxiety_Level": {
            "Low",
            "Medium",
            "High",
        },
    }

    for key, options in allowed.items():

        if categorical[key] not in options:

            raise ValueError(
                f"Invalid value for "
                f"{key.replace('_', ' ')}."
            )

    return {
        **numeric,
        **categorical,
    }


# --------------------------------------------------
# PREDICT PURCHASE LIKELIHOOD
# --------------------------------------------------

def predict_profile(profile):

    row = pd.DataFrame(
        [profile],
        columns=FEATURES,
    )

    prediction = str(
        PIPELINE.predict(row)[0]
    )

    probabilities = (
        PIPELINE.predict_proba(row)[0]
    )

    classes = list(
        PIPELINE.classes_
    )

    probability = float(
        probabilities[
            classes.index("Yes")
        ]
    ) * 100

    return {
        "prediction": prediction,

        "probability": round(
            probability,
            1,
        ),

        "label": (
            "Higher model-estimated likelihood"
            if probability >= 50
            else "Lower model-estimated likelihood"
        ),
    }


# --------------------------------------------------
# EV READINESS SCORE
# --------------------------------------------------

def readiness_score(profile):

    commute = (
        min(
            profile["Daily_Commute_km"] / 60,
            1,
        ) * 15
    )

    home_charging = (
        15
        if profile["Home_Charging_Possible"] == "Yes"
        else 0
    )

    home_stations = (
        min(
            profile[
                "Charging_Stations_Near_Home"
            ] / 10,
            1,
        ) * 15
    )

    work_stations = (
        min(
            profile[
                "Charging_Stations_Near_Work"
            ] / 10,
            1,
        ) * 10
    )

    environmental = (
        (
            profile[
                "Environmental_Concern_Level"
            ] - 1
        ) / 4
    ) * 15

    subsidy = (
        10
        if profile["Subsidy_Available"] == "Yes"
        else 0
    )

    anxiety_points = {
        "Low": 20,
        "Medium": 10,
        "High": 0,
    }

    score = (
        commute
        + home_charging
        + home_stations
        + work_stations
        + environmental
        + subsidy
        + anxiety_points[
            profile["Range_Anxiety_Level"]
        ]
    )

    return int(
        round(
            max(
                0,
                min(100, score),
            )
        )
    )


# --------------------------------------------------
# PERSONALIZED RECOMMENDATIONS
# --------------------------------------------------

def make_recommendations(profile):

    recommendations = []

    if (
        profile["Home_Charging_Possible"]
        == "No"
    ):

        recommendations.append({
            "title": "Explore charging access",

            "text": (
                "Check whether your residence, "
                "workplace, or regular destinations "
                "offer practical charging options."
            ),
        })

    if (
        profile["Range_Anxiety_Level"]
        == "High"
    ):

        recommendations.append({
            "title": "Plan your driving range",

            "text": (
                "Compare your daily commute with "
                "real-world EV range and identify "
                "charging stops for longer trips."
            ),
        })

    if (
        profile["Subsidy_Available"]
        == "No"
    ):

        recommendations.append({
            "title": "Review available incentives",

            "text": (
                "Check official local and national "
                "incentive information before "
                "comparing total ownership costs."
            ),
        })

    if (
        profile["Charging_Stations_Near_Home"] < 3
        and
        profile["Charging_Stations_Near_Work"] < 3
    ):

        recommendations.append({
            "title": "Find nearby public chargers",

            "text": (
                "Explore charging stations near "
                "home, work, and frequently visited "
                "destinations."
            ),
        })

    if (
        profile["Environmental_Concern_Level"]
        <= 2
    ):

        recommendations.append({
            "title": "Compare practical ownership factors",

            "text": (
                "Consider purchase price, running "
                "costs, maintenance, and charging "
                "convenience."
            ),
        })

    if not recommendations:

        recommendations.append({
            "title": "Compare EV options",

            "text": (
                "Compare vehicle price, usable range, "
                "charging speed, warranty, and "
                "ownership costs before deciding."
            ),
        })

    return recommendations


# --------------------------------------------------
# FRONTEND
# --------------------------------------------------

@app.get("/")
def index():

    return render_template(
        "index.html"
    )


# --------------------------------------------------
# DATASET OVERVIEW API
# --------------------------------------------------

@app.get("/api/overview")
def overview():

    return jsonify({
        "summary": read_json(
            SUMMARY_PATH
        ),

        "model_metrics": read_json(
            MODEL_DIR / "model_metrics.json"
        ),
    })


# --------------------------------------------------
# PERSONAL ASSESSMENT API
# --------------------------------------------------

@app.post("/api/predict")
def predict():
    try:
        payload = request.get_json(silent=True) or {}

        profile = clean_payload(payload)
        prediction = predict_profile(profile)

        return jsonify({
            **prediction,
            "readiness_score": readiness_score(profile),
            "recommendations": make_recommendations(profile),
            "profile": profile,
        })

    except (ValueError, TypeError, KeyError) as exc:
        app.logger.warning("Prediction input error: %s", exc)
        return jsonify({"error": str(exc)}), 400

    except Exception as exc:
        app.logger.exception("Unexpected error in /api/predict")
        return jsonify({
            "error": "Prediction failed on the server.",
            "detail": str(exc)
        }), 500


# --------------------------------------------------
# WHAT-IF SIMULATOR API
# --------------------------------------------------

@app.post("/api/simulate")
def simulate():

    try:

        payload = (
            request.get_json(silent=True)
            or {}
        )

        original_profile = clean_payload(
            payload.get("profile", {})
        )

        changed_profile = (
            original_profile.copy()
        )

        changes = payload.get(
            "changes",
            {},
        )

        allowed_changes = {
            "Charging_Stations_Near_Home",
            "Charging_Stations_Near_Work",
            "Home_Charging_Possible",
            "Subsidy_Available",
            "Range_Anxiety_Level",
        }

        for key, value in changes.items():

            if key in allowed_changes:

                changed_profile[key] = value

        changed_profile = clean_payload(
            changed_profile
        )

        original = predict_profile(
            original_profile
        )

        simulated = predict_profile(
            changed_profile
        )

        difference = round(
            simulated["probability"]
            - original["probability"],
            1,
        )

        return jsonify({
            "original": original,

            "simulated": simulated,

            "difference_points": difference,

            "changed_profile": changed_profile,

            "note": (
                "This compares model predictions "
                "under changed inputs. It does not "
                "prove that a change causes an EV purchase."
            ),
        })

    except (
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
    ) as exc:

        return jsonify({
            "error": str(exc)
        }), 400


# --------------------------------------------------
# LOCAL EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    app.run()