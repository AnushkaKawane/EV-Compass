from pathlib import Path
import json
import os

import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    send_from_directory,
)

from model import (
    FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    DATASET_PATH,
    METRICS_PATH,
    get_model,
    predict_probability,
)


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

PUBLIC_DIR = BASE_DIR / "public"

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "ev-compass-development-key"
)


# --------------------------------------------------
# DEFAULT ASSESSMENT
# --------------------------------------------------

DEFAULT_DATA = {
    "Age": 30,
    "Gender": "Other",
    "Annual_Income_USD": 60000,
    "City_Type": "Urban",
    "Daily_Commute_km": 25,
    "Number_of_Cars_Owned": 1,
    "Current_Car_Type": "Sedan",
    "Charging_Stations_Near_Home": 3,
    "Charging_Stations_Near_Work": 3,
    "Home_Charging_Possible": "Yes",
    "Environmental_Concern_Level": 3,
    "Subsidy_Available": "Yes",
    "Range_Anxiety_Level": "Medium",
}


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

MODEL, MODEL_METRICS = get_model()


# --------------------------------------------------
# SERVE FRONTEND ASSETS
# --------------------------------------------------

@app.route("/style.css")
def style_css():
    response = send_from_directory(
        str(PUBLIC_DIR),
        "style.css"
    )
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.route("/script.js")
def script_js():
    response = send_from_directory(
        str(PUBLIC_DIR),
        "script.js"
    )
    response.headers["Cache-Control"] = "no-cache"
    return response


# --------------------------------------------------
# LOAD DATASET FOR INSIGHTS
# --------------------------------------------------

def get_dataset():

    df = pd.read_csv(DATASET_PATH)

    df["Will_Buy_EV"] = (
        df["Will_Buy_EV"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    return df


# --------------------------------------------------
# PARSE USER INPUT
# --------------------------------------------------

def parse_form(form):

    data = {}

    for column in NUMERIC_FEATURES:

        value = form.get(
            column,
            ""
        ).strip()

        if not value:
            raise ValueError(
                f"Please enter {column.replace('_', ' ')}."
            )

        if column in [
            "Age",
            "Number_of_Cars_Owned",
            "Charging_Stations_Near_Home",
            "Charging_Stations_Near_Work",
            "Environmental_Concern_Level",
        ]:
            data[column] = int(value)
        else:
            data[column] = float(value)

    for column in CATEGORICAL_FEATURES:

        value = form.get(
            column,
            ""
        ).strip()

        if not value:
            raise ValueError(
                f"Please select {column.replace('_', ' ')}."
            )

        data[column] = value

    # Validate realistic ranges.
    if not 18 <= data["Age"] <= 100:
        raise ValueError(
            "Age must be between 18 and 100."
        )

    if data["Annual_Income_USD"] < 0:
        raise ValueError(
            "Income cannot be negative."
        )

    if data["Daily_Commute_km"] < 0:
        raise ValueError(
            "Commute distance cannot be negative."
        )

    if data["Number_of_Cars_Owned"] < 0:
        raise ValueError(
            "Number of cars cannot be negative."
        )

    for column in [
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work",
    ]:
        if data[column] < 0:
            raise ValueError(
                "Charging station counts cannot be negative."
            )

    if not 1 <= data["Environmental_Concern_Level"] <= 5:
        raise ValueError(
            "Environmental concern must be between 1 and 5."
        )

    allowed_values = {
        "Gender": [
            "Male", "Female", "Other"
        ],
        "City_Type": [
            "Urban", "Suburban", "Rural"
        ],
        "Current_Car_Type": [
            "Sedan", "SUV", "Hatchback", "Truck"
        ],
        "Home_Charging_Possible": [
            "Yes", "No"
        ],
        "Subsidy_Available": [
            "Yes", "No"
        ],
        "Range_Anxiety_Level": [
            "Low", "Medium", "High"
        ],
    }

    for column, options in allowed_values.items():
        if data[column] not in options:
            raise ValueError(
                f"Invalid selection for {column.replace('_', ' ')}."
            )

    return data


# --------------------------------------------------
# READINESS SCORE
# --------------------------------------------------

def calculate_readiness(data):

    score = 0

    # Home charging accessibility.
    if data["Home_Charging_Possible"] == "Yes":
        score += 25

    # Nearby charging stations.
    home_stations = data[
        "Charging_Stations_Near_Home"
    ]

    work_stations = data[
        "Charging_Stations_Near_Work"
    ]

    score += min(
        home_stations,
        10
    ) * 2

    score += min(
        work_stations,
        10
    ) * 1.5

    # Range anxiety.
    anxiety_scores = {
        "Low": 20,
        "Medium": 12,
        "High": 4,
    }

    score += anxiety_scores[
        data["Range_Anxiety_Level"]
    ]

    # Environmental concern.
    score += (
        data["Environmental_Concern_Level"]
        * 2
    )

    # Daily commute suitability assessment.
    commute = data["Daily_Commute_km"]

    if commute <= 40:
        score += 10
    elif commute <= 80:
        score += 6
    else:
        score += 2

    return max(
        0,
        min(
            100,
            round(score)
        )
    )


# --------------------------------------------------
# PERSONALIZED RECOMMENDATIONS
# --------------------------------------------------

def generate_recommendations(data):

    recommendations = []

    if data["Home_Charging_Possible"] == "No":
        recommendations.append(
            "Explore workplace or public charging options "
            "before deciding whether an EV suits your routine."
        )

    if data["Charging_Stations_Near_Home"] < 3:
        recommendations.append(
            "Check the actual availability and operating hours "
            "of public charging stations near your home."
        )

    if data["Range_Anxiety_Level"] == "High":
        recommendations.append(
            "Compare your daily travel distance with the usable "
            "range of EV models you are considering."
        )

    if data["Daily_Commute_km"] > 80:
        recommendations.append(
            "Investigate charging opportunities along your "
            "regular routes and during longer journeys."
        )

    if data["Subsidy_Available"] == "No":
        recommendations.append(
            "Check whether any current EV incentives apply "
            "in your location."
        )

    if data["Environmental_Concern_Level"] <= 2:
        recommendations.append(
            "Compare energy consumption, charging costs, "
            "and lifecycle environmental impacts when evaluating vehicles."
        )

    if not recommendations:
        recommendations.append(
            "Compare the real-world range, purchase price, "
            "charging costs, and warranty of suitable EV models."
        )

    return recommendations


# --------------------------------------------------
# LANDING PAGE
# --------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# --------------------------------------------------
# ASSESSMENT PAGE
# --------------------------------------------------

@app.route("/assessment")
def assessment():

    return render_template(
        "assessment.html",
        data=session.get(
            "last_assessment",
            DEFAULT_DATA
        ),
        error=None
    )


# --------------------------------------------------
# PROCESS ASSESSMENT
# --------------------------------------------------

@app.route(
    "/assess",
    methods=["POST"]
)
def assess():

    try:

        data = parse_form(
            request.form
        )

        probability = predict_probability(
            MODEL,
            data
        )

        prediction = (
            "Yes"
            if probability >= 0.5
            else "No"
        )

        readiness = calculate_readiness(
            data
        )

        if readiness >= 75:
            readiness_label = "Well Positioned"
        elif readiness >= 50:
            readiness_label = "Getting Ready"
        else:
            readiness_label = "Needs Planning"

        session["last_assessment"] = data

        return render_template(
            "results.html",
            data=data,
            probability=round(
                probability * 100,
                1
            ),
            prediction=prediction,
            readiness=readiness,
            readiness_label=readiness_label,
            recommendations=generate_recommendations(
                data
            )
        )

    except (ValueError, TypeError) as error:

        return render_template(
            "assessment.html",
            data=request.form,
            error=str(error)
        ), 400


# --------------------------------------------------
# WHAT-IF SIMULATOR
# --------------------------------------------------

@app.route(
    "/simulator",
    methods=["GET", "POST"]
)
def simulator():

    baseline = session.get(
        "last_assessment",
        DEFAULT_DATA
    ).copy()

    baseline_probability = (
        predict_probability(
            MODEL,
            baseline
        ) * 100
    )

    scenario_probability = None

    change = None

    error = None

    scenario = baseline.copy()

    if request.method == "POST":

        try:

            scenario["Home_Charging_Possible"] = (
                request.form.get(
                    "Home_Charging_Possible",
                    baseline["Home_Charging_Possible"]
                )
            )

            scenario["Range_Anxiety_Level"] = (
                request.form.get(
                    "Range_Anxiety_Level",
                    baseline["Range_Anxiety_Level"]
                )
            )

            scenario["Subsidy_Available"] = (
                request.form.get(
                    "Subsidy_Available",
                    baseline["Subsidy_Available"]
                )
            )

            scenario[
                "Charging_Stations_Near_Home"
            ] = int(
                request.form.get(
                    "Charging_Stations_Near_Home",
                    baseline["Charging_Stations_Near_Home"]
                )
            )

            if scenario[
                "Charging_Stations_Near_Home"
            ] < 0:
                raise ValueError(
                    "Charging station count cannot be negative."
                )

            if scenario[
                "Home_Charging_Possible"
            ] not in ["Yes", "No"]:
                raise ValueError(
                    "Invalid home charging selection."
                )

            if scenario[
                "Range_Anxiety_Level"
            ] not in ["Low", "Medium", "High"]:
                raise ValueError(
                    "Invalid range anxiety selection."
                )

            if scenario[
                "Subsidy_Available"
            ] not in ["Yes", "No"]:
                raise ValueError(
                    "Invalid subsidy selection."
                )

            scenario_probability = (
                predict_probability(
                    MODEL,
                    scenario
                ) * 100
            )

            change = round(
                scenario_probability
                - baseline_probability,
                1
            )

        except (ValueError, TypeError) as exception:
            error = str(exception)

    return render_template(
        "simulator.html",
        baseline=baseline,
        scenario=scenario,
        baseline_probability=round(
            baseline_probability,
            1
        ),
        scenario_probability=(
            round(
                scenario_probability,
                1
            )
            if scenario_probability is not None
            else None
        ),
        change=change,
        error=error
    )


# --------------------------------------------------
# INSIGHTS DASHBOARD
# --------------------------------------------------

@app.route("/insights")
def insights():

    df = get_dataset()

    total_records = len(df)

    yes_count = (
        df["Will_Buy_EV"]
        .eq("Yes")
        .sum()
    )

    adoption_rate = (
        round(
            yes_count / total_records * 100,
            1
        )
        if total_records
        else 0
    )

    # Adoption by city.
    city_data = (
        df.groupby("City_Type")[
            "Will_Buy_EV"
        ]
        .apply(
            lambda values:
            values.eq("Yes").mean() * 100
        )
        .round(1)
        .to_dict()
    )

    # Adoption by range anxiety.
    anxiety_order = [
        "Low",
        "Medium",
        "High"
    ]

    anxiety_data = (
        df.groupby("Range_Anxiety_Level")[
            "Will_Buy_EV"
        ]
        .apply(
            lambda values:
            values.eq("Yes").mean() * 100
        )
        .round(1)
        .to_dict()
    )

    # Adoption by home charging availability.
    charging_data = (
        df.groupby("Home_Charging_Possible")[
            "Will_Buy_EV"
        ]
        .apply(
            lambda values:
            values.eq("Yes").mean() * 100
        )
        .round(1)
        .to_dict()
    )

    chart_data = {
        "city_labels": list(
            city_data.keys()
        ),
        "city_values": list(
            city_data.values()
        ),
        "anxiety_labels": anxiety_order,
        "anxiety_values": [
            anxiety_data.get(
                level,
                0
            )
            for level in anxiety_order
        ],
        "charging_labels": list(
            charging_data.keys()
        ),
        "charging_values": list(
            charging_data.values()
        ),
    }

    metrics = {}

    if METRICS_PATH.exists():
        try:
            with open(
                METRICS_PATH,
                "r",
                encoding="utf-8"
            ) as file:
                metrics = json.load(file)
        except (OSError, json.JSONDecodeError):
            metrics = {}

    return render_template(
        "insights.html",
        total_records=total_records,
        adoption_rate=adoption_rate,
        yes_count=int(yes_count),
        chart_data=chart_data,
        metrics=metrics,
        scenario={}
    )


# --------------------------------------------------
# LOCAL DEVELOPMENT
# --------------------------------------------------

if __name__ == "__main__":
    app.run()