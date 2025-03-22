import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import mlflow
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

load_dotenv()

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
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
        
        logging.info("MLFLOW_TRACKING_URI: %s", self.mlflow_tracking_uri)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        logging.info("Starting inference pipeline")
        
        # Set the experiment
        mlflow.set_experiment("bank_churn_prediction")
        
        # Validate input parameters
        if not os.path.exists(self.input_data_path):
            raise FileNotFoundError(f"Input data file not found: {self.input_data_path}")
        
        if self.model_run_id is None:
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
                self.model_run_id = runs[0].info.run_id
                logging.info(f"Using latest successful run: {self.model_run_id}")
            else:
                raise ValueError("No successful runs found in the MLflow experiment")
        
        self.next(self.load_model)
    
    @step
    def load_model(self):
        """Load the trained model from MLflow."""
        logging.info(f"Loading model from run ID: {self.model_run_id}")
        
        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            
            # Load the model using MLflow's API
            self.model = mlflow.catboost.load_model(f"runs:/{self.model_run_id}/model")
            logging.info("Model loaded successfully")
                
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            self.model = None
            raise
        
        self.next(self.load_data)
    
    @step
    def load_data(self):
        """Load and prepare the input data for inference."""
        logging.info(f"Loading data from: {self.input_data_path}")
        
        try:
            self.data = pd.read_csv(self.input_data_path)
            logging.info(f"Loaded {len(self.data)} samples for inference")
            
            # Check if the data has the expected columns
            required_columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                               'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
                               'EstimatedSalary', 'Surname']
            
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f"Input data is missing required columns: {missing_columns}")
            
            # Store the original data for later reference
            self.original_data = self.data.copy()
            
            # Check if 'Exited' column exists (for evaluation)
            self.has_labels = 'Exited' in self.data.columns
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise
        
        self.next(self.preprocess_data)
    
    @step
    def preprocess_data(self):
        """Preprocess the input data for inference.
        
        This step applies the same preprocessing steps as used during training.
        """
        logging.info("Preprocessing data for inference")
        
        try:
            # Handle categorical variables
            label_enc_gender = LabelEncoder()
            label_enc_geography = LabelEncoder()
            
            # Fit and transform categorical variables
            # For Gender, we know the categories are typically 'Male' and 'Female'
            self.data["Gender"] = label_enc_gender.fit_transform(self.data["Gender"])
            
            # For Geography, we need to handle potential new categories
            # In a production system, you would load the encoder from the training pipeline
            self.data["Geography"] = label_enc_geography.fit_transform(self.data["Geography"])
            
            # Define a function to hash surnames to a consistent integer value
            def hash_surname(surname):
                # Use md5 to get a consistent hash across platforms
                hash_obj = hashlib.md5(str(surname).encode())
                # Convert first 4 bytes of hash to integer and take modulo 1000 
                # to limit to 0-999 range
                return int(hash_obj.hexdigest()[:8], 16) % 1000
            
            # Use hash encoding for surnames to handle unseen values consistently
            self.data["Surname"] = self.data["Surname"].apply(hash_surname)
            
            # Prepare features for prediction
            if self.has_labels:
                self.X = self.data.drop(columns=['Exited'])
                if 'CustomerId' in self.X.columns:
                    self.X = self.X.drop(columns=['CustomerId'])
                if 'id' in self.X.columns:
                    self.X = self.X.drop(columns=['id'])
                self.y_true = self.data['Exited']
            else:
                self.X = self.data.copy()
                if 'CustomerId' in self.X.columns:
                    self.X = self.X.drop(columns=['CustomerId'])
                if 'id' in self.X.columns:
                    self.X = self.X.drop(columns=['id'])
            
            logging.info("Data preprocessing complete")
            
        except Exception as e:
            logging.error(f"Error preprocessing data: {str(e)}")
            raise
        
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
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            with mlflow.start_run(run_name=f"inference-{current.run_id}"):
                # Log the model run ID used for inference
                mlflow.log_param("model_run_id", self.model_run_id)
                
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