import logging
import os
import sys
import pandas as pd
import mlflow
import numpy as np
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
# Package versions for reproducibility; can be used with Metaflow's @pypi or conda
# when running on remote/batch. See https://docs.metaflow.org/scaling/dependencies/libraries
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
    "setuptools": "75.1.0",
    "xgboost": "2.0.0"
}



@project(name='bank_customer_churn_prediction')
class Training(FlowSpec):
    """Training pipeline.

    This pipeline loads the dataset, trains and evaluates a model to predict a bank customer churn.
    """
    dataset_dir = Parameter(
        "dataset_dir",
        help="Directory containing train.csv files",
        default="data/",
    )
    dataset_file = Parameter(
        "dataset_file",
        help="Optional path to a single dataset file (overrides dataset_dir scan)",
        default="",
    )
    n_splits = Parameter(
        "n_splits",
        help="Number of cross-validation folds",
        default=5,
    )
    seed = Parameter(
        "seed",
        help="Random seed for reproducibility",
        default=42,
    )
    mlflow_tracking_uri = Parameter(
        "mlflow_tracking_uri",
        help="MLflow tracking server URI",
        default="http://127.0.0.1:8080",
    )
    depth = Parameter(
        "depth",
        help="CatBoost tree depth",
        default=4,
    )
    learning_rate = Parameter(
        "learning_rate",
        help="CatBoost learning rate",
        default=0.1,
    )
    subsample = Parameter(
        "subsample",
        help="CatBoost subsample ratio",
        default=0.8,
    )
    l2_leaf_reg = Parameter(
        "l2_leaf_reg",
        help="CatBoost L2 leaf regularization",
        default=1.0,
    )
    model_type = Parameter(
        "model_type",
        help="Model algorithm to train: 'catboost' or 'xgboost'",
        default="catboost",
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
        """Start and prepare the Training pipeline."""

        # Use Parameter as default; env var can override for remote runs. Store in separate
        # attr because Metaflow Parameters are immutable and cannot be reassigned.
        self._mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", self.mlflow_tracking_uri)
        os.environ["MLFLOW_TRACKING_URI"] = self._mlflow_tracking_uri

        logging.info("MLFLOW_TRACKING_URI: %s", self._mlflow_tracking_uri)
        mlflow.set_tracking_uri(self._mlflow_tracking_uri)
        logging.info("Starting pipeline")

        # Set the experiment
        mlflow.set_experiment("bank_churn_prediction")
        
        try:
            # Start a new MLFlow run
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
            logging.info(f"Started MLflow run with ID: {self.mlflow_run_id}")
        except Exception as e:
            message = f"Failed to connect to MLflow server {self._mlflow_tracking_uri}."
            raise RuntimeError(message) from e
        
        _valid_models = ("catboost", "xgboost")
        if self.model_type not in _valid_models:
            raise ValueError(
                f"model_type must be one of {_valid_models}, got '{self.model_type}'"
            )
        self._model_type = self.model_type

        if self._model_type == "xgboost":
            self.training_parameters = {"random_state": int(self.seed)}
        else:
            self.training_parameters = {
                "subsample": float(self.subsample),
                "learning_rate": float(self.learning_rate),
                "l2_leaf_reg": float(self.l2_leaf_reg),
                "depth": int(self.depth),
                "random_seed": int(self.seed),
            }
        
        mlflow.log_params({
            "model_type": self._model_type,
            "dataset_dir": str(self.dataset_dir),
            "dataset_file": str(self.dataset_file) or "(none)",
            "n_splits": int(self.n_splits),
            "seed": int(self.seed),
            **self.training_parameters,
        })
        mlflow.set_tag("model_type", self._model_type)
        mlflow.set_tag("pipeline", "training")
        
        self.next(self.load_dataset)
    

    @step
    def load_dataset(self):
        """Load and prepare the dataset using src.data."""
        from src.data.load_data import load_data
        from src.data.preprocess import preprocess_data

        if self.dataset_file and str(self.dataset_file).strip():
            # Single file override
            resolved = self.dataset_file if os.path.isabs(self.dataset_file) else os.path.join(ROOT, self.dataset_file)
            if not os.path.isfile(resolved):
                raise FileNotFoundError(f"Dataset file not found: {resolved}")
            files_used = [resolved]
            self.raw_data = [load_data(resolved)]
        else:
            # Directory scan
            dataset_dir_abs = os.path.join(ROOT, self.dataset_dir) if not os.path.isabs(self.dataset_dir) else self.dataset_dir
            if not os.path.isdir(dataset_dir_abs):
                dataset_dir_abs = ROOT
            files = [os.path.join(dataset_dir_abs, f) for f in os.listdir(dataset_dir_abs) if f.endswith("train.csv")]
            if not files:
                raise ValueError(f"No dataset files found in directory: {dataset_dir_abs}")
            files_used = files
            self.raw_data = [load_data(f) for f in files]

        combined = pd.concat(self.raw_data, ignore_index=True)
        self.data = preprocess_data(combined)

        if "Exited" not in self.data.columns:
            raise ValueError("Preprocessed data missing required target column 'Exited'.")

        generator = np.random.default_rng(seed=int(self.seed))
        self.data = self.data.sample(frac=1, random_state=generator)

        mlflow.set_tracking_uri(self._mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_param("dataset_files", ",".join(files_used))

        logging.info("Loaded dataset with %d samples from %d file(s)", len(self.data), len(files_used))
        self.next(self.cross_validation)

    @card
    @step
    def cross_validation(self):
        """Generate the indices to split the data for the cross-validation process."""
        from sklearn.model_selection import StratifiedKFold

        kfold = StratifiedKFold(
            n_splits=int(self.n_splits),
            shuffle=True,
            random_state=int(self.seed),
        )
        splits = list(kfold.split(self.data, self.data["Exited"]))
        self.folds = list(enumerate(splits))

        for fold_idx, (train_idx, test_idx) in self.folds:
            if len(train_idx) == 0 or len(test_idx) == 0:
                raise ValueError(
                    f"Fold {fold_idx} has empty train or test set "
                    "(train=%d, test=%d). Check n_splits or data size."
                    % (len(train_idx), len(test_idx))
                )

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
        """Train a model for this fold."""
        logging.info("Training fold %d with %s...", self.fold, self._model_type)

        mlflow.set_tracking_uri(self._mlflow_tracking_uri)
        with (
            mlflow.start_run(run_id=self.mlflow_run_id),
            mlflow.start_run(
                run_name=f"cross-validation-fold-{self.fold}",
                nested=True,
            ) as run,
        ):
            self.mlflow_fold_run_id = run.info.run_id

            if self._model_type == "xgboost":
                from src.models.train_xgb import train_xgb_model
                self.model = train_xgb_model(
                    self.x_train,
                    self.y_train,
                    X_val=self.x_test,
                    y_val=self.y_test,
                    params=self.training_parameters,
                    log_to_mlflow=True,
                    verbose=0,
                )
            else:
                from src.models.train import train_model
                self.model = train_model(
                    self.x_train,
                    self.y_train,
                    X_val=self.x_test,
                    y_val=self.y_test,
                    params=self.training_parameters,
                    log_to_mlflow=True,
                    verbose=0,
                )

        self.next(self.evaluate_fold)

    @card
    @step
    def evaluate_fold(self):
        """Evaluate the model for this fold using src.models.evaluate."""
        from src.models.evaluate import evaluate_model

        logging.info("Evaluating fold %d...", self.fold)

        self.y_pred = self.model.predict(self.x_test)
        mlflow.set_tracking_uri(self._mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_fold_run_id):
            metrics = evaluate_model(self.model, self.x_test, self.y_test, log_to_mlflow=True)
        self.accuracy = metrics["accuracy"]
        self.precision = metrics["precision"]
        self.recall = metrics["recall"]
        self.f1 = metrics["f1"]
        self.roc_auc = metrics.get("roc_auc")

        self.feature_names = list(self.x_test.columns)
        self.feature_importances = self.model.feature_importances_.tolist()

        logging.info(
            "Fold %d - accuracy: %f - precision: %f - recall: %f - f1: %f",
            self.fold, self.accuracy, self.precision, self.recall, self.f1,
        )

        self.next(self.evaluate_model)

    @card(type="blank")
    @step
    def evaluate_model(self, inputs):
        """Evaluate the overall cross-validation process.

        This function averages the score computed for each individual model to
        determine the final model performance.
        """
        from metaflow.cards import Markdown, Table

        # We need access to the `mlflow_run_id` and `mlflow_tracking_uri` artifacts
        # that we set at the start of the flow, but since we are in a join step, we
        # need to merge the artifacts from the incoming branches to make them
        # available.
        import tempfile
        from pathlib import Path

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.merge_artifacts(
            inputs,
            include=["mlflow_run_id", "_mlflow_tracking_uri", "_model_type", "n_splits"],
        )

        accuracies = [i.accuracy for i in inputs]
        precisions = [i.precision for i in inputs]
        recalls = [i.recall for i in inputs]
        f1s = [i.f1 for i in inputs]
        roc_aucs = [i.roc_auc for i in inputs if i.roc_auc is not None]

        self.mean_accuracy = np.mean(accuracies)
        self.mean_precision = np.mean(precisions)
        self.mean_recall = np.mean(recalls)
        self.mean_f1 = np.mean(f1s)

        self.accuracy_std = np.std(accuracies)
        self.precision_std = np.std(precisions)
        self.recall_std = np.std(recalls)
        self.f1_std = np.std(f1s)

        self.mean_roc_auc = np.mean(roc_aucs) if roc_aucs else None
        self.roc_auc_std = np.std(roc_aucs) if roc_aucs else None

        logging.info("Accuracy: %f ±%f", self.mean_accuracy, self.accuracy_std)
        logging.info("Precision: %f ±%f", self.mean_precision, self.precision_std)
        logging.info("Recall: %f ±%f", self.mean_recall, self.recall_std)
        logging.info("F1: %f ±%f", self.mean_f1, self.f1_std)
        if self.mean_roc_auc is not None:
            logging.info("ROC AUC: %f ±%f", self.mean_roc_auc, self.roc_auc_std)

        best_fold = max(inputs, key=lambda x: x.recall)

        self.best_fold_metrics = {
            'fold': best_fold.fold,
            'accuracy': best_fold.accuracy,
            'precision': best_fold.precision,
            'recall': best_fold.recall,
            'f1': best_fold.f1,
            'mlflow_fold_run_id': best_fold.mlflow_fold_run_id,
        }

        parent_metrics = {
            "accuracy": self.mean_accuracy,
            "accuracy_std": self.accuracy_std,
            "precision": self.mean_precision,
            "precision_std": self.precision_std,
            "recall": self.mean_recall,
            "recall_std": self.recall_std,
            "f1": self.mean_f1,
            "f1_std": self.f1_std,
            "best_fold_accuracy": self.best_fold_metrics['accuracy'],
            "best_fold_precision": self.best_fold_metrics['precision'],
            "best_fold_recall": self.best_fold_metrics['recall'],
            "best_fold_f1": self.best_fold_metrics['f1'],
            "best_fold_number": self.best_fold_metrics['fold'],
        }
        if self.mean_roc_auc is not None:
            parent_metrics["roc_auc"] = self.mean_roc_auc
            parent_metrics["roc_auc_std"] = self.roc_auc_std

        mlflow.set_tracking_uri(self._mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_metrics(parent_metrics)
            mlflow.set_tag("best_fold", str(self.best_fold_metrics['fold']))
            mlflow.set_tag(
                "mlflow.note.content",
                f"{self._model_type} | {self.n_splits}-fold CV | "
                f"accuracy={self.mean_accuracy:.4f} | recall={self.mean_recall:.4f} | f1={self.mean_f1:.4f}",
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                # -- Feature importance (best fold) --
                feature_names = np.array(best_fold.feature_names)
                importances = np.array(best_fold.feature_importances)
                sorted_idx = importances.argsort()
                fig, ax = plt.subplots(figsize=(8, max(6, len(feature_names) * 0.35)))
                ax.barh(feature_names[sorted_idx], importances[sorted_idx], color="steelblue")
                ax.set_xlabel("Importance")
                ax.set_title(f"Feature Importance — Best Fold {best_fold.fold} ({self._model_type})")
                fig.tight_layout()
                fig.savefig(tmpdir / "feature_importance.png", dpi=120, bbox_inches="tight")
                plt.close(fig)

                # -- CV summary bar chart with error bars --
                metric_names = ["Accuracy", "Precision", "Recall", "F1"]
                means = [self.mean_accuracy, self.mean_precision, self.mean_recall, self.mean_f1]
                stds = [self.accuracy_std, self.precision_std, self.recall_std, self.f1_std]
                if self.mean_roc_auc is not None:
                    metric_names.append("ROC AUC")
                    means.append(self.mean_roc_auc)
                    stds.append(self.roc_auc_std)

                fig, ax = plt.subplots(figsize=(8, 5))
                bars = ax.bar(metric_names, means, yerr=stds, capsize=5, color="steelblue", alpha=0.85)
                ax.set_ylim(0, 1.05)
                ax.set_ylabel("Score")
                ax.set_title(f"Cross-Validation Summary ({self.n_splits}-fold)")
                for bar, m, s in zip(bars, means, stds):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + s + 0.02,
                        f"{m:.3f}",
                        ha="center", va="bottom", fontsize=9,
                    )
                fig.tight_layout()
                fig.savefig(tmpdir / "cv_summary.png", dpi=120, bbox_inches="tight")
                plt.close(fig)

                # -- Per-fold metrics comparison --
                fold_ids = [i.fold for i in inputs]
                x = np.arange(len(fold_ids))
                width = 0.2
                fig, ax = plt.subplots(figsize=(max(8, len(fold_ids) * 1.8), 5))
                ax.bar(x - 1.5 * width, accuracies, width, label="Accuracy")
                ax.bar(x - 0.5 * width, precisions, width, label="Precision")
                ax.bar(x + 0.5 * width, recalls, width, label="Recall")
                ax.bar(x + 1.5 * width, f1s, width, label="F1")
                ax.set_xticks(x)
                ax.set_xticklabels([f"Fold {f}" for f in fold_ids])
                ax.set_ylim(0, 1.05)
                ax.set_ylabel("Score")
                ax.set_title("Metrics Across Folds")
                ax.legend(loc="lower right")
                fig.tight_layout()
                fig.savefig(tmpdir / "fold_comparison.png", dpi=120, bbox_inches="tight")
                plt.close(fig)

                for png in tmpdir.glob("*.png"):
                    mlflow.log_artifact(str(png), artifact_path="plots")

        table_data = [
            ["Fold", "Accuracy", "Precision", "Recall", "F1"],
            *[
                [str(i.fold), f"{i.accuracy:.4f}", f"{i.precision:.4f}", f"{i.recall:.4f}", f"{i.f1:.4f}"]
                for i in inputs
            ],
        ]
        current.card.append(Markdown("# Cross-Validation Summary"))
        current.card.append(Table(table_data))
        current.card.append(Markdown(
            f"**Mean ± Std:** Accuracy {self.mean_accuracy:.4f} ± {self.accuracy_std:.4f}, "
            f"Precision {self.mean_precision:.4f} ± {self.precision_std:.4f}, "
            f"Recall {self.mean_recall:.4f} ± {self.recall_std:.4f}, "
            f"F1 {self.mean_f1:.4f} ± {self.f1_std:.4f}"
        ))
        current.card.append(Markdown(
            f"**Best fold (by recall):** Fold {self.best_fold_metrics['fold']} — "
            f"accuracy {self.best_fold_metrics['accuracy']:.4f}, "
            f"precision {self.best_fold_metrics['precision']:.4f}, "
            f"recall {self.best_fold_metrics['recall']:.4f}, "
            f"f1 {self.best_fold_metrics['f1']:.4f}"
        ))
        
        # After we finish evaluating the cross-validation process, we can send the flow
        # to the registration step to register where we'll register the final version of
        # the model.
        self.next(self.register_model)

    @card(type="blank")
    @step
    def register_model(self):
        """Register the model in the Model Registry.
        
        This function will register the best model from cross-validation.
        """
        from metaflow.cards import Markdown
        
        logging.info(
            "Best fold (fold %d) - accuracy: %.3f - precision: %.3f - recall: %.3f - f1: %.3f",
            self.best_fold_metrics['fold'],
            self.best_fold_metrics['accuracy'],
            self.best_fold_metrics['precision'],
            self.best_fold_metrics['recall'],
            self.best_fold_metrics['f1'],
        )

        mlflow.set_tracking_uri(self._mlflow_tracking_uri)

        with mlflow.start_run(run_id=self.mlflow_run_id):
            try:
                # Register the best model
                model_version = mlflow.register_model(
                    f"runs:/{self.best_fold_metrics['mlflow_fold_run_id']}/model",
                    "bank_churn_prediction",
                )
                
                # Transition the model to 'Staging'
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name="bank_churn_prediction",
                    version=model_version.version,
                    stage="Staging",
                )
                
                logging.info(
                    "Registered model version %s in Staging stage",
                    model_version.version,
                )

                current.card.append(
                    Markdown(
                        f"# Model Registration\n\n"
                        f"**Best model:** Fold {self.best_fold_metrics['fold']} "
                        f"(recall {self.best_fold_metrics['recall']:.4f})\n\n"
                        f"**Registered as:** bank_churn_prediction version {model_version.version} (Staging)"
                    )
                )
            except Exception as e:  # noqa: BLE001
                logging.warning(
                    "Failed to register model in MLflow Model Registry: %s",
                    e,
                )

                current.card.append(
                    Markdown(
                        "# Model Registration\n\n"
                        f"**Best model:** Fold {self.best_fold_metrics['fold']} "
                        f"(recall {self.best_fold_metrics['recall']:.4f})\n\n"
                        "**Registration:** Skipped – MLflow Model Registry "
                        "not available or incompatible on current tracking "
                        "server.\n"
                    )
                )
        
        # After registering the model, proceed to the end step
        self.next(self.end)

    @step
    def end(self):
        """End the Training pipeline."""
        mlflow.set_tracking_uri(self._mlflow_tracking_uri)
        mlflow.end_run()
        logging.info("The pipeline finished successfully.")



if __name__ == "__main__":
    Training()




