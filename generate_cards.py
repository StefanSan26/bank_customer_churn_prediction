#!/usr/bin/env python3
"""
Script to generate cards for existing Training flow runs
"""

import subprocess
import sys
from metaflow import Flow

def generate_cards_for_run(run_id):
    """Generate cards for all steps in a Training run"""
    
    # Steps that have @card decorator in the Training flow
    card_steps = ['start', 'cross_validation', 'train_fold', 'evaluate_fold', 'evaluate_model', 'register_model']
    
    for step in card_steps:
        try:
            # Try to create card for this step
            pathspec = f"{run_id}/{step}/1"
            print(f"Creating card for {pathspec}...")
            
            result = subprocess.run([
                sys.executable, 'pipelines/training.py', 'card', 'create', pathspec
            ], capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0:
                print(f"✓ Card created for {step}")
            else:
                print(f"✗ Failed to create card for {step}: {result.stderr}")
                
        except Exception as e:
            print(f"✗ Error creating card for {step}: {e}")

def main():
    try:
        # Get the latest Training run
        flow = Flow('Training')
        latest_run = flow.latest_run
        
        print(f"Generating cards for Training run: {latest_run.id}")
        print(f"Run created: {latest_run.created_at}")
        print()
        
        generate_cards_for_run(latest_run.id)
        
        print("\nTo view cards, use:")
        print(f"python pipelines/training.py card view {latest_run.id}/<step_name>/1")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

