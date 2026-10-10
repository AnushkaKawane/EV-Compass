# EV Compass

EV Compass is a machine-learning web application that estimates
electric vehicle purchase likelihood using the supplied EV Adoption
and Range Anxiety dataset.

It includes a personal assessment, readiness indicator,
what-if simulator, recommendations, and dataset visualisations.

## Features

- Personal EV purchase-likelihood prediction
- EV readiness indicator based on a transparent heuristic
- What-if simulator for charging access, subsidies, and range anxiety
- Personalised recommendations based on the selected profile
- Dataset overview and charts
- Responsive navy, lavender, and off-white interface
- Flask backend
- Vercel deployment configuration

## Project Structure

```text
EV-Compass/
│
├── api/
│   └── index.py
│
├── data/
│   └── EV_Adoption_and_Range_Anxiety_Dataset.csv
│
├── models/
│   ├── README.txt
│   ├── ev_model.joblib
│   ├── dataset_summary.json
│   └── model_metrics.json
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│
├── templates/
│   └── index.html
│
├── .gitignore
├── app.py
├── model.py
├── requirements.txt
├── vercel.json
└── README.md