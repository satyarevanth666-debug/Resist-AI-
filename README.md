# ResistAI - Intelligent Antibiotic Resistance Prediction

ResistAI is an AI-powered decision support web app that predicts antibiotic resistance, explains model reasoning, and recommends effective antibiotic options for selected bacteria.

## Features

- Data ingestion from `data/primary.xlsx` and `data/secondary.csv`
- Automatic column standardization to `Bacteria`, `Antibiotic`, `Result`
- Cleaning pipeline (missing values + duplicates removal)
- Label mapping:
  - `Resistant -> 1`
  - `Susceptible -> 0`
  - `Intermediate -> 2`
- Model training and comparison:
  - Logistic Regression
  - Random Forest (main baseline)
  - XGBoost (advanced, if installed)
- Evaluation metrics:
  - Accuracy, Precision, Recall, F1, ROC-AUC
- Explainability:
  - Feature importance
  - SHAP local explanation
- Recommendation engine:
  - Rank antibiotics for a chosen bacteria
  - `Recommended` (low resistance) vs `Avoid` (high resistance)
- Streamlit dashboard with modern dark card UI

## Project Structure

```text
resistAI/
├── data/
│   ├── primary.xlsx
│   └── secondary.csv
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── recommend.py
│   └── visualize.py
├── models/
├── app/
│   └── app.py
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure datasets exist:

- `data/primary.xlsx`
- `data/secondary.csv`

## Train the Model

```bash
python src/train.py
```

This generates:

- `models/model.pkl`
- `models/cleaned_data.csv`
- `models/metrics.json`

## Run the Streamlit App

```bash
streamlit run app/app.py
```

If `models/model.pkl` is missing, the app auto-runs training (assuming data files are present).

## Notes for Hackathon Demo

- Use the Input Panel to select `Bacteria` and `Antibiotic`
- Trigger prediction to view class + confidence
- Review top `Recommended` and `Avoid` antibiotic ranking
- Show interactive heatmap, confusion matrix, class distribution, and feature importance
- Use SHAP panel for local prediction explainability
