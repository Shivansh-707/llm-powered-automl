import json
import pandas as pd
import numpy as np
from datetime import datetime


def get_dataset_signature(df):
    """Create a fingerprint for dataset similarity matching."""
    return {
        "n_rows": len(df),
        "n_features": len(df.columns),
        "n_numeric": len(df.select_dtypes(include=[np.number]).columns),
        "n_categorical": len(df.select_dtypes(include=['object', 'category']).columns),
        "avg_cardinality": round(float(df.nunique().mean()), 2),
        "missing_pct": round(float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100), 2),
    }


def save_experiment(signature, plan, results):
    """Save experiment to memory for future reference."""
    experiment = {
        "timestamp": datetime.now().isoformat(),
        "dataset_signature": signature,
        "plan": plan,
        "results": results,
    }

    try:
        with open("experiment_memory.json", "r") as f:
            memory = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        memory = []

    memory.append(experiment)

    with open("experiment_memory.json", "w") as f:
        json.dump(memory, f, indent=2, default=str)


def load_memory():
    """Load past experiments."""
    try:
        with open("experiment_memory.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
