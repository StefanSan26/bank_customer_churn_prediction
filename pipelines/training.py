import logging
import os
import sys
from pathlib import Path
import pandas as pd
import mlflow
from dotenv import load_dotenv

load_dotenv()

# Project root for src imports
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

# PYTHON = "3.12"
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
class Training(FlowSpec):
    """Training pipeline.

    This pipeline loads the dataset, trains and evaluates a model to predict a bank customer churn.
    """
    dataset_dir = os.getenv("DATASET_DIR", "data/")
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
        """Start and prepare the Training pipeline."""
        import mlflow

        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")

        logging.info("MLFLOW_TRACKING_URI: %s", self.mlflow_tracking_uri)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        logging.info("Starting pipeline")

        # Set the experiment
        mlflow.set_experiment("bank_churn_prediction")
        
        try:
            # Start a new MLFlow run
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
            logging.info(f"Started MLflow run with ID: {self.mlflow_run_id}")
        except Exception as e:
            message = f"Failed to connect to MLflow server {self.mlflow_tracking_uri}."
            raise RuntimeError(message) from e
        
        self.training_parameters = {
            'subsample': 0.8, 
            'learning_rate': 0.1, 
            'l2_leaf_reg': 1, 
            'depth': 4
        }
        
        self.next(self.load_dataset)
    

    @step
    def load_dataset(self):
        """Load and prepare the dataset using src.data."""
        import numpy as np
        from src.data.load_data import load_data
        from src.data.preprocess import preprocess_data

        dataset_dir_abs = os.path.join(ROOT, self.dataset_dir) if not os.path.isabs(self.dataset_dir) else self.dataset_dir
        if not os.path.isdir(dataset_dir_abs):
            dataset_dir_abs = ROOT
        files = [os.path.join(dataset_dir_abs, f) for f in os.listdir(dataset_dir_abs) if f.endswith("train.csv")]

        logging.info("Found %d file(s) in local directory", len(files))
        if not files:
            raise ValueError("No dataset files found in local directory")

        self.raw_data = [load_data(f) for f in files]
        combined = pd.concat(self.raw_data, ignore_index=True)
        self.data = preprocess_data(combined)

        seed = 42
        generator = np.random.default_rng(seed=seed)
        self.data = self.data.sample(frac=1, random_state=generator)

        logging.info("Loaded dataset with %d samples", len(self.data))
        self.next(self.cross_validation)

    @card
    @step
    def cross_validation(self):
        """Generate the indices to split the data for the cross-validation process."""
        from sklearn.model_selection import KFold

        # We are going to use a 5-fold cross-validation process to evaluate the model,
        # so let's set it up. We'll shuffle the data before splitting it into batches.
        kfold = KFold(n_splits=5, shuffle=True)

        # We can now generate the indices to split the dataset into training and test
        # sets. This will return a tuple with the fold number and the training and test
        # indices for each of 5 folds.
        self.folds = list(enumerate(kfold.split(self.data)))

        # We want to transform the data and train a model using each fold, so we'll use
        # `foreach` to run every cross-validation iteration in parallel. Notice how we
        # pass the tuple with the fold number and the indices to next step.
        self.next(self.transform_fold, foreach="folds")

    @step
    def transform_fold(self):
        """Split data for this fold; preprocessing already done in load_dataset via src.data."""
        from src.features.build_features import build_features

        self.fold, (self.train_indices, self.test_indices) = self.input
        logging.info("Transforming fold %d...", self.fold)

        train_df = self.data.iloc[self.train_indices]
        test_df = self.data.iloc[self.test_indices]
        self.x_train, self.y_train = build_features(train_df)
        self.x_test, self.y_test = build_features(test_df)
        if self.y_train is None or self.y_test is None:
            raise ValueError("Exited column missing")
        self.next(self.train_fold)

    @card
    @step
    def train_fold(self):
        """Train a model for this fold using src.models.train."""
        import mlflow
        from src.models.train import train_model

        logging.info("Training fold %d...", self.fold)

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with (
            mlflow.start_run(run_id=self.mlflow_run_id),
            mlflow.start_run(
                run_name=f"cross-validation-fold-{self.fold}",
                nested=True,
            ) as run,
        ):
            self.mlflow_fold_run_id = run.info.run_id
            mlflow.autolog(log_models=True)
            self.model = train_model(
                self.x_train,
                self.y_train,
                params=self.training_parameters,
                log_to_mlflow=True,
                verbose=0,
            )
            mlflow.catboost.log_model(self.model, "model")

        self.next(self.evaluate_fold)

    @card
    @step
    def evaluate_fold(self):
        """Evaluate the model for this fold using src.models.evaluate."""
        import mlflow
        from src.models.evaluate import evaluate_model

        logging.info("Evaluating fold %d...", self.fold)

        metrics = evaluate_model(self.model, self.x_test, self.y_test, log_to_mlflow=False)
        self.accuracy = metrics["accuracy"]
        self.precision = metrics["precision"]
        self.recall = metrics["recall"]
        self.y_pred = self.model.predict(self.x_test)

        logging.info(
            "Fold %d - accuracy: %f - precision: %f - recall: %f",
            self.fold, self.accuracy, self.precision, self.recall,
        )

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_fold_run_id):
            mlflow.log_metrics({"accuracy": self.accuracy, "precision": self.precision, "recall": self.recall})

        self.next(self.evaluate_model)

    @card
    @step
    def evaluate_model(self, inputs):
        """Evaluate the overall cross-validation process.

        This function averages the score computed for each individual model to
        determine the final model performance.
        """
        import mlflow
        import numpy as np

        # We need access to the `mlflow_run_id` and `mlflow_tracking_uri` artifacts
        # that we set at the start of the flow, but since we are in a join step, we
        # need to merge the artifacts from the incoming branches to make them
        # available.
        self.merge_artifacts(inputs, include=["mlflow_run_id", "mlflow_tracking_uri"])

        # Let's calculate the mean and standard deviation of the accuracy and loss from
        # all the cross-validation folds. Notice how we are accumulating these values
        # using the `inputs` parameter provided by Metaflow.
        metrics = {
            'accuracies': [i.accuracy for i in inputs],
            'precisions': [i.precision for i in inputs],
            'recalls': [i.recall for i in inputs]
        }
        
        self.mean_accuracy = np.mean(metrics['accuracies'])
        self.mean_precision = np.mean(metrics['precisions'])
        self.mean_recall = np.mean(metrics['recalls'])
        
        self.accuracy_std = np.std(metrics['accuracies'])
        self.precision_std = np.std(metrics['precisions'])
        self.recall_std = np.std(metrics['recalls'])
        
        logging.info("Accuracy: %f ±%f", self.mean_accuracy, self.accuracy_std)
        logging.info("Precision: %f ±%f", self.mean_precision, self.precision_std)
        logging.info("Recall: %f ±%f", self.mean_recall, self.recall_std)

        # Let's log the model metrics on the parent run.
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_metrics(
                {
                    "cross_validation_accuracy": self.mean_accuracy,
                    "cross_validation_accuracy_std": self.accuracy_std,
                    "cross_validation_precision": self.mean_precision,
                    "cross_validation_precision_std": self.precision_std,
                    "cross_validation_recall": self.mean_recall,
                    "cross_validation_recall_std": self.recall_std,
                },
            )

        # Find the best performing fold based on accuracy
        best_fold = max(inputs, key=lambda x: x.accuracy)
        
        # Store only the specific attributes we need from the best fold
        self.best_fold_metrics = {
            'fold': best_fold.fold,
            'accuracy': best_fold.accuracy,
            'precision': best_fold.precision,
            'recall': best_fold.recall,
            'mlflow_fold_run_id': best_fold.mlflow_fold_run_id
        }
        
        # After we finish evaluating the cross-validation process, we can send the flow
        # to the registration step to register where we'll register the final version of
        # the model.
        self.next(self.register_model)

    @step
    def register_model(self):
        """Register the model in the Model Registry.
        
        This function will register the best model from cross-validation.
        """
        import mlflow
        
        logging.info(
            "Best fold (fold %d) - accuracy: %.3f - precision: %.3f - recall: %.3f",
            self.best_fold_metrics['fold'],
            self.best_fold_metrics['accuracy'],
            self.best_fold_metrics['precision'],
            self.best_fold_metrics['recall']
        )

        # Set up MLflow tracking
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # Log final metrics to the parent run
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_metrics({
                "mean_accuracy": self.mean_accuracy,
                "mean_precision": self.mean_precision,
                "mean_recall": self.mean_recall,
                "best_fold_accuracy": self.best_fold_metrics['accuracy'],
                "best_fold_precision": self.best_fold_metrics['precision'],
                "best_fold_recall": self.best_fold_metrics['recall'],
                "best_fold_number": self.best_fold_metrics['fold']
            })
            
            # Register the best model
            model_version = mlflow.register_model(
                f"runs:/{self.best_fold_metrics['mlflow_fold_run_id']}/model",
                "bank_churn_prediction"
            )
            
            # Transition the model to 'Staging'
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name="bank_churn_prediction",
                version=model_version.version,
                stage="Staging"
            )
            
            logging.info(f"Registered model version {model_version.version} in Staging stage")
        
        # After registering the model, proceed to the end step
        self.next(self.end)

    @step
    def end(self):
        """End the Training pipeline."""
        logging.info("The pipeline finished successfully.")



if __name__ == "__main__":
    Training()




