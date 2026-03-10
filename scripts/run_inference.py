import os
import sys
import pandas as pd
import mlflow
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

load_dotenv()

# Set tracking URI from env (populated by .env) with fallback to the server default.
# Set both os.environ and mlflow.set_tracking_uri so all internal MLflow code paths
# find the same URI regardless of whether they read the env var or the module global.
_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
os.environ["MLFLOW_TRACKING_URI"] = _tracking_uri
mlflow.set_tracking_uri(_tracking_uri)

# Add project root to path so src packages are importable
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.serving.inference import _load_model_by_flavor
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

CHAMPION_URI = "models:/bank_churn_prediction@champion"

# Load data
print("Loading test data...")
raw_data = load_data(os.path.join(_ROOT, "data", "test.csv"))
print(f"Loaded {len(raw_data)} samples")

# Store original rows for output (before any transformation)
original_data = raw_data.copy()
has_labels = "Exited" in raw_data.columns

# Preprocess and build features using the same pipeline as training
print("Preprocessing data...")
processed = preprocess_data(raw_data)
X, y_true = build_features(processed)
if y_true is None:
    has_labels = False

# Load model and make predictions
print(f"Loading champion model from: {CHAMPION_URI}")
try:
    model = _load_model_by_flavor(CHAMPION_URI)
    print("Model loaded successfully")
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]  # Probability of class 1
    
    # Create a DataFrame with the predictions
    predictions_df = original_data.copy()
    predictions_df['predicted_churn'] = y_pred
    predictions_df['churn_probability'] = y_pred_proba
    
    # Save predictions
    output_path = 'data/predictions.csv'
    predictions_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")
    
    # If we have true labels, evaluate the predictions
    if has_labels:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")
        auc = roc_auc_score(y_true, y_pred_proba)

        print(f"Evaluation metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")

        # Log results to MLflow
        with mlflow.start_run(run_name="inference-champion"):
            mlflow.log_metrics({
                "inference_accuracy": accuracy,
                "inference_precision": precision,
                "inference_recall": recall,
                "inference_f1": f1,
                "inference_auc": auc,
            })
            mlflow.log_artifact(output_path)
    
    # Print summary
    print("Inference results summary:")
    print(f"  Total samples: {len(predictions_df)}")
    print(f"  Predicted churn: {predictions_df['predicted_churn'].sum()} ({predictions_df['predicted_churn'].mean()*100:.2f}%)")
    print(f"  Average churn probability: {predictions_df['churn_probability'].mean():.4f}")
    
except Exception as e:
    print(f"Error during inference: {str(e)}")
    raise 