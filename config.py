import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
# Supports both .env file (local) and Streamlit Cloud secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Try Streamlit secrets if env var not found (for Streamlit Cloud deployment)
if not GROQ_API_KEY:
    try:
        import streamlit as st
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass
MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Rate Limit Config (Groq free tier for llama-3.3-70b)
MAX_RPM = 30
MAX_TPM = 12000
MAX_RPD = 1000
MAX_TPD = 100000

# Model Configuration
BASELINE_MODELS = ["lightgbm", "xgboost", "random_forest"]
CV_FOLDS = 5
RANDOM_STATE = 42

# Feature Engineering
MAX_CARDINALITY_ONEHOT = 10
HIGH_CARDINALITY_THRESHOLD = 50
SKEWNESS_THRESHOLD = 1.0

# SHAP Configuration
SHAP_SAMPLE_SIZE = 500  # Sample size for SHAP to keep it fast
SHAP_TOP_FEATURES = 20

# Paths
RESULTS_DIR = "results"
MEMORY_DB = "experiment_memory.json"
