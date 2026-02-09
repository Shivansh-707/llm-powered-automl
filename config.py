import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"

# Model Configuration
BASELINE_MODELS = ["lightgbm", "xgboost", "random_forest"]
CV_FOLDS = 5
RANDOM_STATE = 42

# Feature Engineering
MAX_CARDINALITY_ONEHOT = 10
HIGH_CARDINALITY_THRESHOLD = 50

# Paths
RESULTS_DIR = "results"
MEMORY_DB = "experiment_memory.json"
