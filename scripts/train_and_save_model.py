import os
import pandas as pd
import numpy as np
import hashlib
import mlflow
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle  # Add this import at the top with other imports
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set MLflow tracking URI
logging.info("Setting up MLflow tracking...")
mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("bank_churn_prediction")

# Load data
logging.info("Loading data...")
data = pd.read_csv('data/train.csv')
logging.info(f"Loaded {len(data)} samples")

# Preprocess data
logging.info("Preprocessing data...")
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

# Split data
X = data.drop(columns=['Exited'])
if 'CustomerId' in X.columns:
    X = X.drop(columns=['CustomerId'])
if 'id' in X.columns:
    X = X.drop(columns=['id'])
y = data['Exited']

# Use label encoding for gender and geography
X["Gender"] = label_enc_gender.fit_transform(X["Gender"])
X["Geography"] = label_enc_geography.fit_transform(X["Geography"])
X["Surname"] = X["Surname"].apply(hash_surname)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
logging.info("Training model...")
with mlflow.start_run() as run:
    # Log parameters
    params = {
        'subsample': 0.8, 
        'learning_rate': 0.1, 
        'l2_leaf_reg': 1, 
        'depth': 4
    }
    mlflow.log_params(params)
    
    # Train model
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, verbose=0)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    
    # Log metrics
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    })
    
    # Log model
    mlflow.catboost.log_model(model, "model")
    
    # Save model as pkl file
    model_path = "models/catboost_model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Model saved as pickle file at: {model_path}")
    
    logging.info(f"Model trained and saved with run_id: {run.info.run_id}")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    
    # Save run ID to file for easy reference
    with open("model_run_id.txt", "w") as f:
        f.write(run.info.run_id)