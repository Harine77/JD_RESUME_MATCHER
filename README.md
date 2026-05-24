# JD Resume Matcher

## Problem Statement

Matching candidate resumes to job descriptions (JD) is time-consuming and error-prone when done manually. This project builds an automated matcher that uses classical ML models to predict whether a resume is a good match for a JD.

## Solution Overview

- Train and compare multiple classifiers (Logistic Regression, Naive Bayes, KNN, Decision Tree).
- Build simple ensembles (voting, bagging) and select the best model by evaluation metrics.
- Expose the trained models via a lightweight Flask API so a browser-based frontend can query predictions.

## Tech Stack

- Language: Python
- Web framework: Flask
- Tunneling (for demo): pyngrok
- Data & ML: pandas, numpy, scikit-learn
- Frontend: A static HTML page is included (`jd_resume_matcher_pro.html`) that can call the Flask API

## What I Achieved

- Implemented and compared several classical ML classifiers for resume-JD matching.
- Built a simple Voting ensemble and a Bagging fallback.
- Exposed endpoints: `/predict`, `/metrics`, and `/health` via Flask for easy integration with a frontend.

## Metrics

The project stores per-model metrics (accuracy, precision, recall, F1) during training and exposes them at the `/metrics` endpoint. Replace the placeholders below with your trained results (or call the endpoint after starting the backend).

- Best ensemble: see `BEST_ENSEMBLE_INFO` in the backend.
- Example reported score behaviour in code: predicted `score` may be returned as numeric (e.g. 78 for a match).

Please run the notebook training cells to populate `ALL_MODELS_METRICS` and `BEST_ENSEMBLE_INFO` before starting the backend.

## API Endpoints

- `GET /health` — health check
- `POST /predict` — main prediction endpoint. Request JSON:

```json
{
  "resume": "...",
  "jd": "...",
  "model": "lr|nb|knn|dt|voting|bagging"
}
```

Response contains `verdict`, `score`, `model`, `model_predictions`, `all_models_metrics`, and `best_ensemble`.

- `GET /metrics` — returns stored model metrics and best ensemble info

## How to Run (Colab / Notebook workflow)

1. Train your models in the notebook cells as in `JD_Resume_Matcher_ML_Updated.ipynb`.
2. Add the `flask_backend_code.py` cell (or import/run it) after training so `TRAINED_MODELS`, `ALL_MODELS_METRICS`, and `BEST_ENSEMBLE_INFO` are populated.
3. Install dependencies and start the backend (example for a notebook):

```bash
pip install flask pyngrok scikit-learn pandas numpy
# Then run the backend cell; the code will start an ngrok tunnel and print a public URL.
```

4. Point the frontend (`jd_resume_matcher_pro.html`) to the ngrok public URL + `/predict`.

## How to Push This Project to GitHub

1. Create an empty repository on GitHub (choose `main` as default branch).
2. From the project root run:

```bash
git init
git add .
git commit -m "Initial commit: add project files and README"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

If you prefer SSH, use `git@github.com:<your-username>/<your-repo>.git` as the remote URL.

Note: I cannot push directly to your GitHub without your credentials. If you want me to prepare a script or help run the push locally, tell me which step you'd like help with.

## Files of Interest

- `flask_backend_code.py` — Flask backend exposing `/predict` and `/metrics`.
- `JD_Resume_Matcher_ML_Updated.ipynb` — Notebook with training and evaluation steps.
- `jd_resume_matcher_pro.html` — Example frontend that calls the API.

## Next Steps & Suggestions

- Replace metric placeholders with numbers from your notebook or call `/metrics` after starting the backend.
- Optionally create a `requirements.txt` and CI workflow to run tests or linting.
- If you want a GitHub repository created and the code pushed automatically, provide a repository URL and grant limited access (or run the push commands above locally).

---

If you'd like, I can also:
- generate a `requirements.txt` from the environment,
- create a simple `Procfile` for deployment, or
- help craft a concise Git commit history before you push.
