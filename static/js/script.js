const state = {
    profile: null,
    lastResult: null,
    simulatorChart: null,
    targetChart: null,
    cityChart: null
};

const pageNames = {
    overview: "Overview",
    assessment: "Personal assessment",
    simulator: "What-if simulator",
    recommendations: "Recommendations"
};


// ==========================================
// TOAST MESSAGES
// ==========================================

function showToast(message) {
    const toast = document.getElementById("toast");

    toast.textContent = message;
    toast.classList.add("show");

    clearTimeout(showToast.timer);

    showToast.timer = setTimeout(() => {
        toast.classList.remove("show");
    }, 3500);
}


// ==========================================
// TAB NAVIGATION
// ==========================================

function openTab(tab) {
    document.querySelectorAll(".page").forEach(page => {
        page.classList.remove("active");
    });

    const page = document.getElementById(`page-${tab}`);

    if (page) {
        page.classList.add("active");
    }

    document.querySelectorAll(".nav-item").forEach(button => {
        button.classList.toggle(
            "active",
            button.dataset.tabLink === tab
        );
    });

    document.getElementById("breadcrumbCurrent").textContent =
        pageNames[tab] || "Overview";

    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
}


document.addEventListener("click", event => {
    const button = event.target.closest("[data-tab-link]");

    if (button) {
        event.preventDefault();
        openTab(button.dataset.tabLink);
    }
});


// ==========================================
// FORM DATA
// ==========================================

function getFormProfile(form) {
    const data = new FormData(form);
    const result = {};

    for (const [key, value] of data.entries()) {
        result[key] = value;
    }

    return result;
}


// ==========================================
// API REQUESTS
// ==========================================

async function postJSON(url, body) {
    const response = await fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(body)
    });

    const data = await response.json();

    if (!response.ok) {
        throw new Error(
            data.error || "Something went wrong. Please try again."
        );
    }

    return data;
}


function setText(id, value) {
    const element = document.getElementById(id);

    if (element) {
        element.textContent = value;
    }
}


// ==========================================
// RECOMMENDATIONS
// ==========================================

function renderRecommendations(recommendations) {
    const container = document.getElementById("recommendationList");

    container.innerHTML = "";

    recommendations.forEach((item, index) => {
        const card = document.createElement("article");
        card.className = "panel recommendation-card";

        const number = document.createElement("span");
        number.className = "recommendation-number";
        number.textContent = String(index + 1).padStart(2, "0");

        const body = document.createElement("div");

        const title = document.createElement("h3");
        title.textContent = item.title;

        const paragraph = document.createElement("p");
        paragraph.textContent = item.text;

        body.append(title, paragraph);

        const arrow = document.createElement("span");
        arrow.className = "recommendation-arrow";
        arrow.textContent = "↗";

        card.append(number, body, arrow);

        container.appendChild(card);
    });
}


// ==========================================
// DISPLAY ASSESSMENT RESULT
// ==========================================

function renderAssessment(result) {
    document
        .getElementById("assessmentResult")
        .classList.remove("hidden");

    setText(
        "predictionHeading",
        result.prediction === "Yes"
            ? "EV purchase: Yes"
            : "EV purchase: No"
    );

    setText(
        "predictionText",
        `${result.label}. This is an estimate produced by the trained dataset model.`
    );

    setText(
        "probabilityValue",
        `${result.probability.toFixed(1)}%`
    );

    document.getElementById("probabilityBar").style.width =
        `${Math.max(0, Math.min(100, result.probability))}%`;

    setText("readinessValue", result.readiness_score);

    setText(
        "recommendationProfileTitle",
        result.prediction === "Yes"
            ? "Your model leans toward EV adoption"
            : "Explore the barriers in your profile"
    );

    setText(
        "recommendationSummaryText",
        `The model estimates a ${result.probability.toFixed(1)}% probability of a “Yes” label for the inputs you provided. The readiness indicator is a separate heuristic.`
    );

    setText(
        "recommendationScore",
        `${result.readiness_score}/100`
    );

    renderRecommendations(result.recommendations);

    document
        .getElementById("assessmentResult")
        .scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });
}


// ==========================================
// ASSESSMENT FORM SUBMISSION
// ==========================================

document
    .getElementById("assessmentForm")
    .addEventListener("submit", async event => {

        event.preventDefault();

        const button = document.getElementById("predictButton");
        const original = button.innerHTML;

        button.disabled = true;
        button.textContent = "Calculating...";

        try {
            const profile = getFormProfile(event.currentTarget);

            const result = await postJSON(
                "/api/predict",
                profile
            );

            state.profile = result.profile;
            state.lastResult = result;

            renderAssessment(result);

            syncSimulatorFromProfile(result.profile);

            showToast(
                "Your profile has been assessed. You can now test what-if scenarios."
            );

        } catch (error) {
            showToast(error.message);

        } finally {
            button.disabled = false;
            button.innerHTML = original;
        }
    });


// ==========================================
// SIMULATOR PROFILE
// ==========================================

function syncSimulatorFromProfile(profile) {
    document.getElementById("homeStations").value =
        profile.Charging_Stations_Near_Home;

    document.getElementById("workStations").value =
        profile.Charging_Stations_Near_Work;

    document.getElementById("simHomeCharging").value =
        profile.Home_Charging_Possible;

    document.getElementById("simSubsidy").value =
        profile.Subsidy_Available;

    document.getElementById("simAnxiety").value =
        profile.Range_Anxiety_Level;

    updateSliderLabels();
}


function updateSliderLabels() {
    setText(
        "homeStationsValue",
        document.getElementById("homeStations").value
    );

    setText(
        "workStationsValue",
        document.getElementById("workStations").value
    );
}


["homeStations", "workStations"].forEach(id => {
    document.getElementById(id).addEventListener(
        "input",
        updateSliderLabels
    );
});


// ==========================================
// OPEN SIMULATOR FROM ASSESSMENT
// ==========================================

document
    .getElementById("sendToSimulator")
    .addEventListener("click", () => {

        if (!state.profile) {
            showToast("Complete your assessment first.");
            return;
        }

        openTab("simulator");
    });


// ==========================================
// RUN WHAT-IF SIMULATION
// ==========================================

document
    .getElementById("simulatorForm")
    .addEventListener("submit", async event => {

        event.preventDefault();

        if (!state.profile) {
            showToast(
                "Complete your personal assessment first so the simulator has a baseline."
            );

            openTab("assessment");
            return;
        }

        const changes = {
            Charging_Stations_Near_Home: Number(
                document.getElementById("homeStations").value
            ),

            Charging_Stations_Near_Work: Number(
                document.getElementById("workStations").value
            ),

            Home_Charging_Possible:
                document.getElementById("simHomeCharging").value,

            Subsidy_Available:
                document.getElementById("simSubsidy").value,

            Range_Anxiety_Level:
                document.getElementById("simAnxiety").value
        };

        const button = event.currentTarget.querySelector(
            "button[type='submit']"
        );

        const original = button.innerHTML;

        button.disabled = true;
        button.textContent = "Comparing...";

        try {
            const result = await postJSON(
                "/api/simulate",
                {
                    profile: state.profile,
                    changes: changes
                }
            );

            setText(
                "originalProbability",
                `${result.original.probability.toFixed(1)}%`
            );

            setText(
                "simulatedProbability",
                `${result.simulated.probability.toFixed(1)}%`
            );

            const difference = result.difference_points;

            setText(
                "differenceText",
                `${difference > 0 ? "+" : ""}${difference.toFixed(1)} percentage points`
            );

            setText("simulationNote", result.note);

            renderSimulationChart(
                result.original.probability,
                result.simulated.probability
            );

        } catch (error) {
            showToast(error.message);

        } finally {
            button.disabled = false;
            button.innerHTML = original;
        }
    });


// ==========================================
// SIMULATION CHART
// ==========================================

function renderSimulationChart(original, simulated) {
    const context = document.getElementById("simulationChart");

    if (!window.Chart) {
        showToast("Chart library could not load.");
        return;
    }

    if (state.simulatorChart) {
        state.simulatorChart.destroy();
    }

    state.simulatorChart = new Chart(context, {
        type: "bar",

        data: {
            labels: [
                "Original profile",
                "What-if scenario"
            ],

            datasets: [
                {
                    label: "Estimated probability (%)",
                    data: [original, simulated],

                    backgroundColor: [
                        "#b9b0fb",
                        "#8172ed"
                    ],

                    borderRadius: 7,
                    maxBarThickness: 70
                }
            ]
        },

        options: {
            responsive: true,
            maintainAspectRatio: false,

            plugins: {
                legend: {
                    display: false
                },

                tooltip: {
                    callbacks: {
                        label: item => `${item.raw.toFixed(1)}%`
                    }
                }
            },

            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,

                    grid: {
                        color: "#eff0f6"
                    },

                    ticks: {
                        callback: value => `${value}%`,
                        font: {
                            size: 10
                        },
                        color: "#9297aa"
                    },

                    border: {
                        display: false
                    }
                },

                x: {
                    grid: {
                        display: false
                    },

                    ticks: {
                        color: "#6f748a",
                        font: {
                            size: 10
                        }
                    },

                    border: {
                        display: false
                    }
                }
            }
        }
    });
}


// ==========================================
// LOAD OVERVIEW DATA
// ==========================================

async function loadOverview() {
    try {
        const response = await fetch("/api/overview");

        if (!response.ok) {
            throw new Error("Could not load overview.");
        }

        const data = await response.json();
        const summary = data.summary || {};

        setText(
            "recordCount",
            Number(summary.rows || 0).toLocaleString()
        );

        setText(
            "targetName",
            summary.target || "Will_Buy_EV"
        );

        setText(
            "buyRate",
            summary.buy_rate_percent == null
                ? "—"
                : `${summary.buy_rate_percent}%`
        );

        const accuracy =
            data.model_metrics &&
            data.model_metrics.accuracy;

        setText(
            "modelAccuracy",
            accuracy == null
                ? "—"
                : `${(accuracy * 100).toFixed(1)}%`
        );


        // Purchase-intention chart

        const targetCounts = summary.target_counts || {};

        makeDoughnut(
            "targetChart",
            ["Yes", "No"],
            [
                targetCounts.Yes || 0,
                targetCounts.No || 0
            ],
            [
                "#8172ed",
                "#dcd8fb"
            ]
        );


        // City-type chart

        const cityCounts = summary.city_type_counts || {};

        makeBar(
            "cityChart",
            Object.keys(cityCounts),
            Object.values(cityCounts),
            "#9a8cf5"
        );

    } catch (error) {
        showToast(
            "Dataset overview could not load. Check that the model was trained and the dataset is present."
        );
    }
}


// ==========================================
// DOUGHNUT CHART
// ==========================================

function makeDoughnut(id, labels, values, colors) {
    const canvas = document.getElementById(id);

    if (!canvas || !window.Chart) {
        return;
    }

    if (state.targetChart) {
        state.targetChart.destroy();
    }

    state.targetChart = new Chart(canvas, {
        type: "doughnut",

        data: {
            labels: labels,

            datasets: [
                {
                    data: values,
                    backgroundColor: colors,
                    borderWidth: 0,
                    hoverOffset: 5
                }
            ]
        },

        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: "69%",

            plugins: {
                legend: {
                    position: "bottom",

                    labels: {
                        usePointStyle: true,
                        pointStyle: "circle",
                        boxWidth: 7,
                        padding: 18,
                        color: "#6f748a",

                        font: {
                            size: 10
                        }
                    }
                }
            }
        }
    });
}


// ==========================================
// BAR CHART
// ==========================================

function makeBar(id, labels, values, color) {
    const canvas = document.getElementById(id);

    if (!canvas || !window.Chart) {
        return;
    }

    if (state.cityChart) {
        state.cityChart.destroy();
    }

    state.cityChart = new Chart(canvas, {
        type: "bar",

        data: {
            labels: labels,

            datasets: [
                {
                    data: values,
                    backgroundColor: color,
                    borderRadius: 6,
                    maxBarThickness: 42
                }
            ]
        },

        options: {
            responsive: true,
            maintainAspectRatio: false,

            plugins: {
                legend: {
                    display: false
                },

                tooltip: {
                    callbacks: {
                        label: item =>
                            `${item.raw.toLocaleString()} records`
                    }
                }
            },

            scales: {
                y: {
                    beginAtZero: true,

                    grid: {
                        color: "#eff0f6"
                    },

                    ticks: {
                        color: "#9297aa",

                        font: {
                            size: 9
                        }
                    },

                    border: {
                        display: false
                    }
                },

                x: {
                    grid: {
                        display: false
                    },

                    ticks: {
                        color: "#6f748a",

                        font: {
                            size: 10
                        }
                    },

                    border: {
                        display: false
                    }
                }
            }
        }
    });
}


// ==========================================
// INITIALISE
// ==========================================

loadOverview();
updateSliderLabels();