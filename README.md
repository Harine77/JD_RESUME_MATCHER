# JD Resume Matcher

Automatically match candidate resumes to job descriptions (JDs) using classical machine learning models, with easy web-based access for predictions and metrics.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Flask](https://img.shields.io/badge/Flask-web-lightgrey) ![ML](https://img.shields.io/badge/ML-scikit--learn-yellow)

---

## Problem

Screening resumes for the right job fit is tedious and error-prone if done manually. The JD Resume Matcher automates this by leveraging machine learning models to predict candidate fit, helping streamline recruitment pipelines.

## Solution Features

- Multiple ML classifiers: Logistic Regression, Naive Bayes, KNN, Decision Tree
- Ensemble models (Voting, Bagging) for improved prediction
- Interactive web API via Flask
- Simple HTML frontend to demo predictions
- Model metrics tracking and serving

## Tech Stack

- Languages: Python (main), Jupyter Notebook (experimentation), HTML (frontend)
- ML/Data: scikit-learn, pandas, numpy
- Web: Flask, pyngrok (for tunneling/public links)
- Frontend: `jd_resume_matcher_pro.html` (calls backend API)

## Usage

### 1. Train & Evaluate Models

- Open `JD_Resume_Matcher_ML_Updated.ipynb` in Jupyter/Colab
- Run cells to:
  - Prepare data
  - Train all classifiers & ensembles
  - Store metrics in `ALL_MODELS_METRICS`, `BEST_ENSEMBLE_INFO`

### 2. Start the Backend

Install dependencies and run the backend (in notebook or locally):

```bash
pip install flask pyngrok scikit-learn pandas numpy
# Run: See Flask backend code in `flask_backend_code.py`
```

The backend starts an ngrok tunnel and prints a public URL.

### 3. Use the Frontend

Edit `jd_resume_matcher_pro.html` to point to your backend’s `/predict` endpoint (ngrok URL).

---

## API Reference

- `GET /health` – Check server status
- `POST /predict` – Predict match

    **Request:**
    ```json
    {
      "resume": "...",
      "jd": "...",
      "model": "lr|nb|knn|dt|voting|bagging"
    }
    ```
    **Response:** Includes `verdict`, confidence `score`, `model`, and per-model predictions/metrics

- `GET /metrics` – List model performance & best ensemble

---

## Metrics & Model Selection

After training, metrics such as accuracy, precision, recall and F1 for all models are available at `/metrics`. The best ensemble (“Voting” or “Bagging”) can be used for highest accuracy predictions.

---

## Project Structure

- `JD_Resume_Matcher_ML_Updated.ipynb` — Notebook for data prep and model training
- `flask_backend_code.py` — Flask app serving predictions and metrics
- `jd_resume_matcher_pro.html` — Minimal frontend UI
- Add a `requirements.txt` and `Procfile` for deployment

---

> JD Resume Matcher – Quickly find the best-fit candidates for your jobs, every time.
