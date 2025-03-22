#!/usr/bin/env python
"""
Example script to demonstrate how to use the inference pipeline.
This script runs the inference pipeline on a sample dataset.
"""

import os
import sys
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Run the inference pipeline on a sample dataset."""
    # Check if the input file exists
    input_file = "data/train.csv"  # Using the training data as an example
    if not os.path.exists(input_file):
        logging.error(f"Input file not found: {input_file}")
        return 1
    
    # Set the output file path
    output_file = "data/example_predictions.csv"
    
    # Run the inference pipeline
    logging.info(f"Running inference pipeline on {input_file}")
    logging.info(f"Results will be saved to {output_file}")
    
    try:
        # Build the command to run the inference pipeline
        cmd = [
            "python", 
            "pipelines/inference.py", 
            "run", 
            f"--input_data_path={input_file}", 
            f"--output_data_path={output_file}"
        ]
        
        # Run the command
        logging.info(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # Check if the output file was created
        if os.path.exists(output_file):
            logging.info(f"Inference completed successfully. Results saved to {output_file}")
            return 0
        else:
            logging.error(f"Output file not found: {output_file}")
            return 1
    
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running inference pipeline: {e}")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 