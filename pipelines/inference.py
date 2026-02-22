import logging
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from metaflow import (
    FlowSpec,
    Parameter,
    card,
    current,
    environment,
    project,
    step,
)

# Using the same package dependencies as the training pipeline
PACKAGES = {
    "azure-ai-ml": "1.19.0",
    "azureml-mlflow": "1.57.0.post1",
    "boto3": "1.35.32",
    "catboost": "1.2.7",
    "evidently": "0.4.33",
    "jax[cpu]": "0.4.24",
    "keras": "2.15.0",
    "mlflow": "2.17.1",
    "numpy": "1.26.4",
    "packaging": "24.1",
    "pandas": "2.2.3",
    "python-dotenv": "1.0.1",
    "requests": "2.32.3",
    "scikit-learn": "1.5.2",
    "setuptools": "75.1.0"
}

@project(name='bank_customer_churn_prediction')
class Inference(FlowSpec):
    """Inference pipeline.

    This pipeline loads a trained model from MLflow and makes predictions on new data.
    """
    
    # Parameters that can be passed when running the pipeline
    input_data_path = Parameter(
        "input_data_path",
        help="Path to the input data file for inference",
        default="data/test.csv"
    )
    
    output_data_path = Parameter(
        "output_data_path",
        help="Path to save the prediction results",
        default="data/predictions.csv"
    )
    
    model_run_id = Parameter(
        "model_run_id",
        help="MLflow run ID of the model to use for inference",
        default=None
    )
    
    logging.basicConfig(level=logging.INFO)
    
    @card
    @environment(
        vars={
            "MLFLOW_TRACKING_URI": os.getenv(
                "MLFLOW_TRACKING_URI",
                "http://127.0.0.1:8080",
            ),
        },
    )
    @step
    def start(self):
        """Start and prepare the Inference pipeline."""
        # Log working directory
        logging.info(f"Current working directory: {os.getcwd()}")
        
        # Store values in instance variables that are meant to be modified
        self._mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
        self._model_run_id = self.model_run_id  # Store parameter value in instance variable
        
        logging.info("MLFLOW_TRACKING_URI: %s", self._mlflow_tracking_uri)
        mlflow.set_tracking_uri(self._mlflow_tracking_uri)
        logging.info("Starting inference pipeline")
        
        # Set the experiment
        mlflow.set_experiment("bank_churn_prediction")
        
        # Validate input parameters
        if not os.path.exists(self.input_data_path):
            raise FileNotFoundError(f"Input data file not found: {self.input_data_path}")
        
        if self._model_run_id is None:
            # If no specific run ID is provided, try to get the latest successful run
            logging.info("No model run ID provided, attempting to find the latest successful run")
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name("bank_churn_prediction")
            if experiment is None:
                raise ValueError("Could not find the bank_churn_prediction experiment")
                
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="attributes.status = 'FINISHED'",
                order_by=["attributes.start_time DESC"],
                max_results=1
            )
            
            if runs:
                self._model_run_id = runs[0].info.run_id
                logging.info(f"Using latest successful run: {self._model_run_id}")
            else:
                raise ValueError("No successful runs found in the MLflow experiment")
        
        self.next(self.load_model)
    
    @step
    def load_model(self):
        """Load the trained model from MLflow Model Registry."""
        logging.info("Loading model from Model Registry: models:/bank_churn_prediction/Staging")
        
        try:
            mlflow.set_tracking_uri(self._mlflow_tracking_uri)
            
            # Load the model directly from the Model Registry
            logging.info("Attempting to load model from Model Registry...")
            self.model = mlflow.catboost.load_model("models:/bank_churn_prediction/Staging")
            
            # Log model feature names
            feature_names = self.model.feature_names_
            logging.info(f"Model feature names: {feature_names}")
            
            logging.info("Model loaded successfully from Model Registry")
                
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            logging.error(f"MLflow tracking URI: {self._mlflow_tracking_uri}")
            logging.error(f"Current working directory: {os.getcwd()}")
            self.model = None
            raise
        
        self.next(self.load_data)
    
    @step
    def load_data(self):
        """Load input data using src.data.load_data."""
        from src.data.load_data import load_data as load_data_fn

        logging.info("Loading data from: %s", self.input_data_path)
        path = self.input_data_path if os.path.isabs(self.input_data_path) else os.path.join(ROOT, self.input_data_path)
        self.data = load_data_fn(path)
        logging.info("Loaded %d samples for inference", len(self.data))

        required_columns = [
            "CreditScore", "Geography", "Gender", "Age", "Tenure",
            "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
            "EstimatedSalary", "Surname",
        ]
        missing_columns = [c for c in required_columns if c not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Input data missing required columns: {missing_columns}")

        self.original_data = self.data.copy()
        self.has_labels = "Exited" in self.data.columns
        self.next(self.preprocess_data)
    
    @step
    def preprocess_data(self):
        """Preprocess using src.data.preprocess_data and build feature matrix for model."""
        from src.data.preprocess import preprocess_data
        from src.features.build_features import build_features

        logging.info("Preprocessing data for inference")
        self.data = preprocess_data(self.data)

        if self.has_labels:
            self.X, self.y_true = build_features(self.data)
        else:
            self.X, _ = build_features(self.data)
            self.y_true = None

        if self.X is None:
            raise ValueError("Could not build feature matrix")

        # Align columns to model order (add id/CustomerId if missing)
        if "id" not in self.X.columns:
            self.X["id"] = self.X.index
        if "CustomerId" not in self.X.columns:
            self.X["CustomerId"] = self.X.get("id", self.X.index).astype(str)
        expected_order = [
            "id", "CustomerId", "Surname", "CreditScore", "Geography", "Gender",
            "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
            "IsActiveMember", "EstimatedSalary",
        ]
        missing_cols = [c for c in expected_order if c not in self.X.columns]
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
        self.X = self.X[expected_order]
        logging.info("Data preprocessing complete")
        self.next(self.make_predictions)
    
    @card
    @step
    def make_predictions(self):
        """Make predictions using the loaded model."""
        logging.info("Making predictions")
        
        try:
            if self.model is None:
                raise ValueError("No model available for inference")
            
            # Make predictions
            self.y_pred = self.model.predict(self.X)
            self.y_pred_proba = self.model.predict_proba(self.X)[:, 1]  # Probability of class 1
            
            # Create a DataFrame with the predictions
            self.predictions_df = self.original_data.copy()
            self.predictions_df['predicted_churn'] = self.y_pred
            self.predictions_df['churn_probability'] = self.y_pred_proba
            
            logging.info(f"Made predictions for {len(self.predictions_df)} samples")
            
            # If we have true labels, evaluate the predictions
            if self.has_labels:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
                
                self.accuracy = accuracy_score(self.y_true, self.y_pred)
                self.precision = precision_score(self.y_true, self.y_pred, average="macro")
                self.recall = recall_score(self.y_true, self.y_pred, average="macro")
                self.auc = roc_auc_score(self.y_true, self.y_pred_proba)
                
                logging.info(f"Evaluation metrics:")
                logging.info(f"  Accuracy: {self.accuracy:.4f}")
                logging.info(f"  Precision: {self.precision:.4f}")
                logging.info(f"  Recall: {self.recall:.4f}")
                logging.info(f"  AUC: {self.auc:.4f}")
            
        except Exception as e:
            logging.error(f"Error making predictions: {str(e)}")
            raise
        
        self.next(self.save_results)
    
    @step
    def save_results(self):
        """Save the prediction results."""
        logging.info(f"Saving prediction results to: {self.output_data_path}")
        
        try:
            # Create the output directory if it doesn't exist
            output_dir = os.path.dirname(self.output_data_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Save the predictions
            self.predictions_df.to_csv(self.output_data_path, index=False)
            logging.info(f"Saved predictions to {self.output_data_path}")
            
            # Log the results to MLflow
            mlflow.set_tracking_uri(self._mlflow_tracking_uri)
            with mlflow.start_run(run_name=f"inference-{current.run_id}"):
                # Log the model run ID used for inference
                mlflow.log_param("model_run_id", self._model_run_id)
                
                # Log metrics if available
                if self.has_labels:
                    mlflow.log_metrics({
                        "inference_accuracy": self.accuracy,
                        "inference_precision": self.precision,
                        "inference_recall": self.recall,
                        "inference_auc": self.auc
                    })
                
                # Log the predictions file as an artifact
                mlflow.log_artifact(self.output_data_path)
                
                # Log additional information
                mlflow.log_param("input_data_path", self.input_data_path)
                mlflow.log_param("num_samples", len(self.predictions_df))
                
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
            raise
        
        self.next(self.end)
    
    @step
    def end(self):
        """End the Inference pipeline."""
        logging.info("Inference pipeline completed successfully")
        
        # Print a summary of the results
        if hasattr(self, 'has_labels') and self.has_labels:
            logging.info("Inference results summary:")
            logging.info(f"  Total samples: {len(self.predictions_df)}")
            logging.info(f"  Predicted churn: {self.predictions_df['predicted_churn'].sum()} ({self.predictions_df['predicted_churn'].mean()*100:.2f}%)")
            logging.info(f"  Accuracy: {self.accuracy:.4f}")
            logging.info(f"  Precision: {self.precision:.4f}")
            logging.info(f"  Recall: {self.recall:.4f}")
            logging.info(f"  AUC: {self.auc:.4f}")
        else:
            logging.info("Inference results summary:")
            logging.info(f"  Total samples: {len(self.predictions_df)}")
            logging.info(f"  Predicted churn: {self.predictions_df['predicted_churn'].sum()} ({self.predictions_df['predicted_churn'].mean()*100:.2f}%)")
            logging.info(f"  Average churn probability: {self.predictions_df['churn_probability'].mean():.4f}")


if __name__ == "__main__":
    Inference() 