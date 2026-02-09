import json
import pandas as pd
import numpy as np
from datetime import datetime

def get_dataset_signature(df):
    """Create a simple signature for dataset similarity matching."""
    return {
        "n_rows": len(df),
        "n_features": len(df.columns),
        "n_numeric": len(df.select_dtypes(include=[np.number]).columns),
        "n_categorical": len(df.select_dtypes(include=['object', 'category']).columns),
        "avg_cardinality": df.nunique().mean()
    }

def save_experiment(signature, plan, results):
    """Save experiment to memory for future reference."""
    experiment = {
        "timestamp": datetime.now().isoformat(),
        "dataset_signature": signature,
        "plan": plan,
        "results": results
    }
    
    try:
        with open("experiment_memory.json", "r") as f:
            memory = json.load(f)
    except FileNotFoundError:
        memory = []
    
    memory.append(experiment)
    
    with open("experiment_memory.json", "w") as f:
        json.dump(memory, f, indent=2)

def load_memory():
    """Load past experiments."""
    try:
        with open("experiment_memory.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
