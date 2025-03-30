    # Removed model_run_id parameter as we're now using the Model Registry
        
        logging.info("MLFLOW_TRACKING_URI: %s", self._mlflow_tracking_uri)
        mlflow.set_tracking_uri(self._mlflow_tracking_uri)
        logging.info("Starting inference pipeline")
        
        # Set the experiment
        mlflow.set_experiment("bank_churn_prediction")
        
        # Validate input parameters
        if not os.path.exists(self.input_data_path):
            raise FileNotFoundError(f"Input data file not found: {self.input_data_path}")
