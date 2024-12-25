import logging
import os
from pathlib import Path
import pandas as pd


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
    
    @step
    def start(self):
        """Start and prepare the Training pipeline."""
        logging.info("Starting pipeline")
        print('starting pipeline')

        self.next(self.load_dataset)

    @step
    def load_dataset(self):
        """Load and prepare the dataset.
        """
        import numpy as np

        files = [os.path.join(self.dataset_dir, f) for f in os.listdir(self.dataset_dir) if f.endswith('.csv')]

        logging.info("Found %d file(s) in local directory", len(files))

        raw_data = [pd.read_csv(file) for file in files]
        data = pd.concat(raw_data, ignore_index=True)

        # Replace extraneous values in the sex column with NaN. We can handle missing
        # values later in the pipeline.
        data.dropna(inplace=True)
        data.reset_index(drop=True,inplace=True)

        # We want to shuffle the dataset. We can use the current time as the seed to ensure a # different shuffle each time the pipeline is executed.
        seed = 42
        generator = np.random.default_rng(seed=seed)
        data = data.sample(frac=1, random_state=generator)

        logging.info("Loaded dataset with %d samples", len(data))
        self.next(self.end)

    @step
    def end(self):
        """End the Training pipeline."""
        logging.info("The pipeline finished successfully.")



if __name__ == "__main__":
    Training()