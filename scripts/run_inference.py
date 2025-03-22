import os
import pandas as pd
import numpy as np
import hashlib
import mlflow
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("bank_churn_prediction")

# Get run ID from file or use provided one
try:
    with open("model_run_id.txt", "r") as f:
        run_id = f.read().strip()
    print(f"Using run ID from file: {run_id}")
except FileNotFoundError:
    # Use the latest run ID from the training pipeline
    run_id = "20c87e3a77ea42ec80cfa58a5d2971b7"
    print(f"Using provided run ID: {run_id}")

# Load data
print("Loading test data...")
data = pd.read_csv('data/test.csv')
print(f"Loaded {len(data)} samples")

# Preprocess data
print("Preprocessing data...")
# Handle categorical variables
label_enc_gender = LabelEncoder()
label_enc_geography = LabelEncoder()

# Define a function to hash surnames to a consistent integer value
def hash_surname(surname):
    # Use md5 to get a consistent hash across platforms
    hash_obj = hashlib.md5(str(surname).encode())
    # Convert first 4 bytes of hash to integer and take modulo 1000 
    # to limit to 0-999 range
    return int(hash_obj.hexdigest()[:8], 16) % 1000

# Store original data for output
original_data = data.copy()

# Check if 'Exited' column exists (for evaluation)
has_labels = 'Exited' in data.columns

# Prepare features
if has_labels:
    X = data.drop(columns=['Exited'])
    if 'CustomerId' in X.columns:
        X = X.drop(columns=['CustomerId'])
    if 'id' in X.columns:
        X = X.drop(columns=['id'])
    y_true = data['Exited']
else:
    X = data.copy()
    if 'CustomerId' in X.columns:
        X = X.drop(columns=['CustomerId'])
    if 'id' in X.columns:
        X = X.drop(columns=['id'])

# Use label encoding for gender and geography
X["Gender"] = label_enc_gender.fit_transform(X["Gender"])
X["Geography"] = label_enc_geography.fit_transform(X["Geography"])
X["Surname"] = X["Surname"].apply(hash_surname)

# Load model and make predictions
print(f"Loading model from run ID: {run_id}")
try:
    # Load the model using MLflow's API
    model = mlflow.catboost.load_model(f"runs:/{run_id}/model")
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
        
        print(f"Evaluation metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        
        # Log results to MLflow
        with mlflow.start_run(run_name=f"inference-{run_id}"):
            mlflow.log_metrics({
                "inference_accuracy": accuracy,
                "inference_precision": precision,
                "inference_recall": recall
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