# Bank Customer Churn Prediction

This project implements a machine learning solution to predict customer churn for a bank. It uses historical customer data to train a model that can identify customers who are likely to leave the bank, enabling proactive retention strategies.

## Project Overview

The project uses CatBoost, a gradient boosting algorithm, to predict customer churn based on various features such as credit score, age, balance, and more. The implementation includes:

- Data preprocessing and feature engineering
- Model training with cross-validation
- Model evaluation and performance metrics
- Inference pipeline for making predictions on new data
- MLflow integration for experiment tracking and model management
- Metaflow pipelines for orchestration and reproducibility

## Project Structure

- `data/`: Contains the dataset files
  - `train.csv`: Training dataset
  - `test.csv`: Test dataset for evaluation
  - `predictions.csv`: Output file with churn predictions
- `pipelines/`: Contains the Metaflow pipelines
  - `training.py`: Pipeline for training the churn prediction model
  - `inference.py`: Pipeline for making predictions using a trained model
- `scripts/`: Contains standalone scripts
  - `train_and_save_model.py`: Direct script for training and saving a model
  - `run_inference.py`: Direct script for making predictions
  - `split_data.py`: Script for splitting data into train and test sets

## Setup

1. Install `uv` (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate
```

3. Install the required dependencies:
```bash
uv pip install -r requirements.txt
```

4. Set up MLflow tracking server with SQLite backend for persistence:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5001
```

5. Create the MLflow experiment:
```bash
mlflow experiments create -n "bank_churn_prediction"
```

6. Set environment variables (or create a `.env` file):
```
MLFLOW_TRACKING_URI=http://127.0.0.1:5001
DATASET_DIR=data/
```


## Training the Model

### Using Metaflow Pipeline

The training pipeline loads the dataset, performs preprocessing, trains a CatBoost model using cross-validation, and logs the results to MLflow.

```bash
python -m pipelines.training run
```

### Using Direct Script

For a simpler approach without Metaflow:

```bash
python scripts/train_and_save_model.py
```

This script will:
- Load and preprocess the data
- Train a CatBoost model
- Log the model and metrics to MLflow
- Save the model run ID to `model_run_id.txt` for later use

## Making Predictions

### Using Metaflow Pipeline

```bash
# Using the latest trained model
python -m pipelines.inference run --input_data_path data/test.csv --output_data_path data/metaflow_predictions.csv

# Using a specific model run ID
python -m pipelines.inference run --input_data_path data/test.csv --output_data_path data/metaflow_predictions.csv --model_run_id YOUR_MLFLOW_RUN_ID
```

### Using Direct Script

For a simpler approach without Metaflow:

```bash
python scripts/run_inference.py
```

This script will:
- Load the model run ID from `model_run_id.txt` (or use a default)
- Load and preprocess the test data
- Make predictions and save them to `data/predictions.csv`
- Log the inference results to MLflow

## Model Features

The model uses the following features to predict customer churn:

- **CreditScore**: Customer's credit score
- **Geography**: Customer's location (France, Spain, Germany)
- **Gender**: Customer's gender
- **Age**: Customer's age
- **Tenure**: Number of years the customer has been with the bank
- **Balance**: Customer's account balance
- **NumOfProducts**: Number of bank products the customer uses
- **HasCrCard**: Whether the customer has a credit card (1=Yes, 0=No)
- **IsActiveMember**: Whether the customer is an active member (1=Yes, 0=No)
- **EstimatedSalary**: Customer's estimated salary
- **Surname**: Customer's surname (encoded using hashing)

## Model Performance

The model achieves the following performance metrics on the test dataset:

- **Accuracy**: ~86.85%
- **Precision**: ~82.21%
- **Recall**: ~75.45%
- **AUC**: ~89.25%

These metrics indicate that the model is effective at identifying customers who are likely to churn.

## Viewing Results

You can view the training and inference results in the MLflow UI:

1. Open a web browser and go to http://127.0.0.1:5001
2. Navigate to the "bank_churn_prediction" experiment to see the training and inference runs
3. Click on a run to view detailed metrics, parameters, and artifacts

## Interpreting Predictions

The prediction output includes:

- **predicted_churn**: Binary prediction (1=Will churn, 0=Will not churn)
- **churn_probability**: Probability of churn (between 0 and 1)

A higher churn probability indicates a higher risk of the customer leaving the bank.

## Troubleshooting

If you encounter issues with MLflow:

1. Ensure the MLflow server is running
2. Check that the MLFLOW_TRACKING_URI environment variable is set correctly
3. Verify that the "bank_churn_prediction" experiment exists

If you encounter issues with the Metaflow pipelines:

1. Check that all dependencies are installed
2. Ensure that the data files exist in the expected locations
3. Try running the direct scripts as an alternative


