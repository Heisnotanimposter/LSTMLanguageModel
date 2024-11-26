# monitoring.py

import time
import pandas as pd

def log_user_interaction(user_id, video_id, action):
    """
    Log user interaction.
    """
    # Append interaction to log (in practice, write to a database or file)
    interaction = {'user_id': user_id, 'video_id': video_id, 'action': action, 'timestamp': time.time()}
    print("Logged Interaction:", interaction)
    # For demonstration, we'll just print it.

def compute_metrics(logs):
    """
    Compute performance metrics.
    """
    df = pd.DataFrame(logs)
    ctr = df[df['action'] == 'click'].shape[0] / df.shape[0]
    average_watch_time = df[df['action'] == 'watch']['duration'].mean()
    print(f"CTR: {ctr}, Average Watch Time: {average_watch_time}")

def check_retraining_needed(metrics_history):
    """
    Determine if model retraining is needed.
    """
    if len(metrics_history) < 2:
        return False
    # Simple check: if CTR drops by more than 10%
    ctr_change = (metrics_history[-1]['ctr'] - metrics_history[-2]['ctr']) / metrics_history[-2]['ctr']
    if ctr_change < -0.1:
        return True
    return False

if __name__ == "__main__":
    # Simulate logging
    logs = []
    for i in range(10):
        log_user_interaction(user_id=1, video_id=101 + i, action='click')
        time.sleep(1)
    
    # Compute metrics
    compute_metrics(logs)
    
    # Check if retraining is needed
    metrics_history = [{'ctr': 0.15}, {'ctr': 0.12}]
    retrain = check_retraining_needed(metrics_history)
    if retrain:
        print("Retraining the model...")
        # Call training function from training.py
    else:
        print("Model performance is stable.")