"""
=================================================================
FLASK BACKEND FOR JD RESUME MATCHER
=================================================================
Add this cell to your Google Colab notebook AFTER training all models.
It exposes a Flask API that your HTML frontend can call.

STEPS:
1. Copy this entire code block
2. Create a NEW CELL in your notebook AFTER the model training cells
3. Paste this code
4. Run the cell
5. When ngrok tunnel URL appears, copy it to your frontend
"""

# --- Install Flask and ngrok ---
# Uncomment these lines if running for the first time
# !pip install flask pyngrok -q

from flask import Flask, request, jsonify
from pyngrok import ngrok
import json

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    """Allow browser frontend to call this API from any origin."""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, ngrok-skip-browser-warning'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# ─────────────────────────────────────────────────────────────
# GLOBAL VARIABLES: Store trained models and metrics
# ─────────────────────────────────────────────────────────────
TRAINED_MODELS = {}
ALL_MODELS_METRICS = {}
BEST_ENSEMBLE_INFO = {}


def init_models_and_metrics():
    """
    Initialize global variables with trained models and metrics.
    This function should be called BEFORE starting Flask.
    It assumes all models are trained in the notebook cells above.
    """
    global TRAINED_MODELS, ALL_MODELS_METRICS, BEST_ENSEMBLE_INFO

    # Store all trained models
    TRAINED_MODELS = {
        'lr': lr_model,           # Logistic Regression
        'nb': nb,                 # Naive Bayes
        'knn': knn,               # KNN
        'dt': dt,                 # Decision Tree
        'voting': None,           # Will be handled in predict function
        'bagging': None           # Will be handled in predict function
    }

    # Store metrics for all models (from the comparison_df created during training)
    # If comparison_df doesn't exist, create it from results dictionary
    try:
        metrics_df = comparison_df  # This should exist from your model comparison cell
    except:
        # Fallback: build from results dictionary if comparison_df not available
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics.get('Accuracy', 0),
                'Precision': metrics.get('Precision', 0),
                'Recall': metrics.get('Recall', 0),
                'F1 Score': metrics.get('F1 Score', 0),
            })
        metrics_df = pd.DataFrame(comparison_data)

    # Convert metrics to dictionary format
    for idx, row in metrics_df.iterrows():
        model_name = row['Model']
        ALL_MODELS_METRICS[model_name] = {
            'accuracy': row.get('Accuracy (%)', row.get('Accuracy', 0)),
            'precision': row.get('Precision (%)', row.get('Precision', 0)),
            'recall': row.get('Recall (%)', row.get('Recall', 0)),
            'f1_score': row.get('F1 Score (%)', row.get('F1 Score', 0)),
        }

    # Find best ensemble (highest F1 score)
    best_f1 = 0
    best_name = 'Voting Ensemble'
    best_desc = 'Combines votes from all 4 base models. Majority wins.'
    best_acc = 0
    best_prec = 0
    best_rec = 0

    for model_name, metrics in ALL_MODELS_METRICS.items():
        if 'Ensemble' in model_name and metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_name = model_name
            best_acc = metrics['accuracy']
            best_prec = metrics['precision']
            best_rec = metrics['recall']
            
            if 'Voting' in model_name:
                best_desc = 'Voting classifier: all 4 models vote, majority decides.'
            elif 'Bagging' in model_name:
                best_desc = '5 decision trees trained on bootstrap samples. Low variance.'

    BEST_ENSEMBLE_INFO = {
        'name': best_name,
        'description': best_desc,
        'f1_score': best_f1,
        'accuracy': best_acc,
        'precision': best_prec,
        'recall': best_rec,
    }

    print('✓ Models and metrics initialized successfully!')
    print(f'✓ All model metrics loaded: {len(ALL_MODELS_METRICS)} models')
    print(f'✓ Best ensemble: {BEST_ENSEMBLE_INFO["name"]} (F1: {BEST_ENSEMBLE_INFO["f1_score"]:.1f}%)')


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'message': 'Backend is running'}), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    
    Request JSON:
    {
        "resume": "resume text here...",
        "jd": "job description text here...",
        "model": "lr" or "nb" or "knn" or "dt" or "voting" or "bagging"
    }
    
    Response JSON:
    {
        "verdict": "MATCH" or "NO MATCH",
        "score": 75,
        "model": "lr",
        "all_models_metrics": { ... },
        "best_ensemble": { ... }
    }
    """
    try:
        data = request.get_json(silent=True) or {}
        resume = data.get('resume', '').strip()
        jd = data.get('jd', '').strip()
        model_choice = data.get('model', 'lr')

        if not resume or not jd:
            return jsonify({'error': 'Resume and JD are required'}), 400

        # Combine text (same as training)
        combined = resume + ' ' + jd

        # Convert to TF-IDF vector
        vector = vectorizer.transform([combined])

        # Get predictions from all models for the current input
        model_predictions = {
            'Naive Bayes': int(nb.predict(vector)[0]),
            'Logistic Regression': int(lr_model.predict(vector)[0]),
            'KNN': int(knn.predict(vector)[0]),
            'Decision Tree': int(dt.predict(vector)[0]),
        }

        # Bagging fallback: use DT until a real bagging model is wired in
        model_predictions['Bagging Ensemble'] = int(dt.predict(vector)[0])

        # Strict majority vote for the ensemble
        match_votes = sum(model_predictions[name] for name in [
            'Naive Bayes', 'Logistic Regression', 'KNN', 'Decision Tree'
        ])
        no_match_votes = 4 - match_votes
        voting_pred = 1 if match_votes > no_match_votes else 0
        model_predictions['Voting Ensemble'] = int(voting_pred)

        # Final result is based on the selected model
        if model_choice == 'lr':
            pred = model_predictions['Logistic Regression']
        elif model_choice == 'nb':
            pred = model_predictions['Naive Bayes']
        elif model_choice == 'knn':
            pred = model_predictions['KNN']
        elif model_choice == 'dt':
            pred = model_predictions['Decision Tree']
        elif model_choice == 'voting':
            pred = model_predictions['Voting Ensemble']
        elif model_choice == 'bagging':
            pred = model_predictions['Bagging Ensemble']
        else:
            pred = model_predictions['Logistic Regression']

        verdict = 'MATCH' if pred == 1 else 'NO MATCH'
        final_result = {
            'model': {
                'lr': 'Logistic Regression',
                'nb': 'Naive Bayes',
                'knn': 'KNN',
                'dt': 'Decision Tree',
                'voting': 'Voting Ensemble',
                'bagging': 'Bagging Ensemble'
            }.get(model_choice, 'Logistic Regression'),
            'predicted_label': pred,
            'verdict': verdict,
            'score': 78 if pred == 1 else 25,
        }

        # Create response
        response = {
            'verdict': verdict,
            'score': final_result['score'],
            'model': model_choice,
            'model_name': final_result['model'],
            'model_predictions': model_predictions,
            'votes': {
                'Naive Bayes': model_predictions['Naive Bayes'],
                'Logistic Regression': model_predictions['Logistic Regression'],
                'KNN': model_predictions['KNN'],
                'Decision Tree': model_predictions['Decision Tree'],
                'Bagging Ensemble': model_predictions['Bagging Ensemble'],
                'Voting Ensemble': model_predictions['Voting Ensemble'],
                'final (majority vote)': model_predictions['Voting Ensemble'],
            },
            'final_result': final_result,
            'all_models_metrics': ALL_MODELS_METRICS,
            'best_ensemble': BEST_ENSEMBLE_INFO,
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get all model metrics."""
    return jsonify({
        'all_models': ALL_MODELS_METRICS,
        'best_ensemble': BEST_ENSEMBLE_INFO
    }), 200


# ─────────────────────────────────────────────────────────────
# MAIN: Start Flask with ngrok
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('\n' + '='*60)
    print('🚀 Starting JD Resume Matcher Flask Backend')
    print('='*60)

    # Initialize models and metrics
    init_models_and_metrics()

    # Start ngrok tunnel
    print('\n🌐 Starting ngrok tunnel...')
    public_tunnel = ngrok.connect(5000)
    public_url = public_tunnel.public_url
    print(f'✓ Public URL: {public_url}')
    print(f'\n📋 Copy this URL to your frontend:')
    print(f'   {public_url}')
    print('\n' + '='*60)

    # Start Flask app
    app.run(port=5000, debug=False)
