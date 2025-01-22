import logging
import os
from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier
import mlflow
from dotenv import load_dotenv


load_dotenv()

from metaflow import (
    FlowSpec,
    Parameter,
    card,
    current,
    environment,
    project,
    pypi_base,
    resources,
    step,
)

# PYTHON = "3.12"
# PACKAGES = {
#     "scikit-learn": "1.5.2",
#     "pandas": "2.2.3",
#     "numpy": "2.1.1",
#     "keras": "3.5.0",
#     "jax[cpu]": "0.4.33",
#     "boto3": "1.35.32",
#     "packaging": "24.1",
#     "mlflow": "2.17.1",
#     "setuptools": "75.1.0",
#     "requests": "2.32.3",
#     "evidently": "0.4.33",
#     "azure-ai-ml": "1.19.0",
#     "azureml-mlflow": "1.57.0.post1",
#     "python-dotenv": "1.0.1",
# }

# def packages(*names: str):
#     """Return a dictionary of the specified packages and their version.

#     This function is useful to set up the different pipelines while keeping the
#     package versions consistent and centralized in a single location.
#     """
#     return {name: PACKAGES[name] for name in names if name in PACKAGES}


@project(name='bank_customer_churn_prediction')
# @pypi_base(
#     python=PYTHON,
#     packages=PACKAGES
# )



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
                "http://127.0.0.1:5000",
            ),
        },
    )
    @step
    def start(self):
        """Start and prepare the Training pipeline."""
        import mlflow

        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

        logging.info("MLFLOW_TRACKING_URI: %s", self.mlflow_tracking_uri)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        logging.info("Starting pipeline")

        # self.next(self.load_dataset)
        # self.training_parameters = {
        #     "epochs": TRAINING_EPOCHS,
        #     "batch_size": TRAINING_BATCH_SIZE,
        # }
        try:
            # Let's start a new MLFlow run to track everything that happens during the
            # execution of this flow. We want to set the name of the MLFlow
            # experiment to the Metaflow run identifier so we can easily
            # recognize which experiment corresponds with each run.
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
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
        """Load and prepare the dataset.
        """
        import numpy as np

        files = [os.path.join(self.dataset_dir, f) for f in os.listdir(self.dataset_dir) if f.endswith('.csv')]

        logging.info("Found %d file(s) in local directory", len(files))
        if not files:
            raise ValueError("No dataset files found in local directory")

        self.raw_data = [pd.read_csv(file) for file in files]
        self.data = pd.concat(self.raw_data, ignore_index=True)

        # Replace extraneous values in the sex column with NaN. We can handle missing
        # values later in the pipeline.
        self.data.dropna(inplace=True)
        self.data.reset_index(drop=True,inplace=True)

        # We want to shuffle the dataset. We can use the current time as the seed to ensure a # different shuffle each time the pipeline is executed.
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
        """Transform the data to build a model during the cross-validation process.

        This step will run for each fold in the cross-validation process. It uses
        a SciKit-Learn pipeline to preprocess the dataset before training a model.
        """
        from sklearn.preprocessing import LabelEncoder
        import hashlib

        # Let's start by unpacking the indices representing the training and test data
        # for the current fold. We computed these values in the previous step and passed
        # them as the input to this step.
        self.fold, (self.train_indices, self.test_indices) = self.input

        logging.info("Transforming fold %d...", self.fold)


        # Finally, let's build the SciKit-Learn pipeline to process the feature columns,
        # fit it to the training data and transform both the training and test data.
        self.x_train = self.data.iloc[self.train_indices]
        self.x_test = self.data.iloc[self.test_indices]
        self.y_train = self.data.iloc[self.train_indices].Exited
        self.y_test = self.data.iloc[self.test_indices].Exited

        label_enc = LabelEncoder()

        # Define a function to hash surnames to a consistent integer value
        def hash_surname(surname):
            # Use md5 to get a consistent hash across platforms
            hash_obj = hashlib.md5(str(surname).encode())
            # Convert first 4 bytes of hash to integer and take modulo 1000 
            # to limit to 0-999 range
            return int(hash_obj.hexdigest()[:8], 16) % 1000

        # Handle categorical variables
        # Use label encoding for gender and geography which have limited categories
        self.x_train["Gender"] = label_enc.fit_transform(self.x_train[["Gender"]])
        self.x_test["Gender"] = label_enc.transform(self.x_test[["Gender"]])
        self.x_train["Geography"] = label_enc.fit_transform(self.x_train[["Geography"]])
        self.x_test["Geography"] = label_enc.transform(self.x_test[["Geography"]])

        # Use hash encoding for surnames to handle unseen values consistently
        self.x_train["Surname"] = self.x_train["Surname"].apply(hash_surname)
        self.x_test["Surname"] = self.x_test["Surname"].apply(hash_surname)


        # After processing the data and storing it as artifacts in the flow, we want
        # to train a model.
        self.next(self.train_fold)

    @card
    @step
    def train_fold(self):
        """Train a model as part of the cross-validation process.

        This step will run for each fold in the cross-validation process. It trains the
        model using the data we processed in the previous step.
        """
        import mlflow

        logging.info("Training fold %d...", self.fold)

        # Let's track the training process under the same experiment we started at the
        # beginning of the flow. Since we are running cross-validation, we can create
        # a nested run for each fold to keep track of each separate model individually.
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with (
            mlflow.start_run(run_id=self.mlflow_run_id),
            mlflow.start_run(
                run_name=f"cross-validation-fold-{self.fold}",
                nested=True,
            ) as run,
        ):
            # Let's store the identifier of the nested run in an artifact so we can
            # reuse it later when we evaluate the model from this fold.
            self.mlflow_fold_run_id = run.info.run_id

            # Let's configure the autologging for the training process. Since we are
            # training the model corresponding to one of the folds, we won't log the
            # model itself.
            mlflow.autolog(log_models=False)

            # Let's now build and fit the model on the training data. Notice how we are
            # using the training data we processed and stored as artifacts in the
            # `transform` step.
            # self.saved_params_catboost = {'subsample': 0.8, 'learning_rate': 0.1, 'l2_leaf_reg': 1, 'depth': 4}
            self.model = CatBoostClassifier(**self.training_parameters)
            # self.model.fit(X=X_train, y=y_train)


            # self.model = build_model(self.x_train.shape[1])
            self.model.fit(
                self.x_train,
                self.y_train,
                verbose=0,
            )
        # After training a model for this fold, we want to evaluate it.
        self.next(self.evaluate_fold)
        # self.next(self.end)

    @card
    @step
    def evaluate_fold(self):
        """Evaluate the model we created as part of the cross-validation process.

        This step will run for each fold in the cross-validation process. It evaluates
        the model using the test data for this fold.
        """
        import mlflow
        from sklearn.metrics import precision_score, recall_score, accuracy_score
        import numpy as np

        logging.info("Evaluating fold %d...", self.fold)

        # Let's evaluate the model using the test data we processed and stored as
        # artifacts during the `transform` step.
        # self.loss, self.accuracy = self.model.evaluate(
        #     self.x_test,
        #     self.y_test,
        #     verbose=2,
        # )
        # Assuming you have your test data (X_test, y_test)
        self.y_pred = self.model.predict(self.x_test)
        
        # Calculate accuracy
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        # Calculate macro-averaged  precision and recall
        self.precision = precision_score(self.y_test, self.y_pred, average="macro")
        self.recall = recall_score(self.y_test, self.y_pred, average="macro")
        #calculate auc
        # self.auc = roc_auc_score(self.y_test, self.y_pred)


        logging.info(
            "Fold %d - accuracy: %f - precision: %f - recall: %f",
            self.accuracy, 
            self.fold,
            self.precision,
            self.recall
        )

        # Let's log everything under the same nested run we created when training the
        # current fold's model.
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_fold_run_id):
            mlflow.log_metrics(
                {
                    "accuracy": self.accuracy,
                    "precision": self.precision,
                    "recall": self.recall,         
                },
            )

        # When we finish evaluating every fold in the cross-validation process, we want
        # to evaluate the overall performance of the model by averaging the scores from
        # each fold.
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

        # After we finish evaluating the cross-validation process, we can send the flow
        # to the registration step to register where we'll register the final version of
        # the model.
        self.next(self.register_model)

    @step
    def register_model(self):
        """Register the model in the Model Registry.
        
        This function will aggregate results from all folds and register the best model.
        """
        import numpy as np
        import mlflow
        
        # All metrics were already calculated in evaluate_model step
        logging.info(
            "Model evaluation complete. Final metrics:"
        )
        logging.info(
            "Mean accuracy: %.3f (±%.3f)",
            self.mean_accuracy,
            self.accuracy_std
        )

        # Find best performing fold based on accuracy
        # self.best_fold = max(inputs, key=lambda x: x.accuracy)

        logging.info(
            "Best fold (fold %d) - accuracy: %.3f - precision: %.3f - recall: %.3f",
            self.best_fold.fold,
            self.best_fold.accuracy,
            self.best_fold.precision,
            self.best_fold.recall
        )

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_metrics({
                "mean_accuracy": self.mean_accuracy,
                "mean_precision": self.mean_precision,
                "mean_recall": self.mean_recall,
                "best_fold_accuracy": self.best_fold.accuracy,
                "best_fold_precision": self.best_fold.precision,
                "best_fold_recall": self.best_fold.recall
            })
        
        # After logging metrics, proceed to the end step
        self.next(self.end)

    @step
    def end(self):
        """End the Training pipeline."""
        logging.info("The pipeline finished successfully.")



if __name__ == "__main__":
    Training()




