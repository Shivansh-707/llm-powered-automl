# 🤖 LLM-Powered AutoML System

An intelligent automated machine learning system that uses **Llama 3.3 70B** (via Groq) to analyze datasets, generate expert-level feature engineering strategies with natural language reasoning, and train baseline models — all with SHAP-based explainability.

> Traditional AutoML uses hardcoded rules. This system uses LLM reasoning to make intelligent, context-aware decisions — and explains every choice it makes.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Expert Profiling** | Correlation analysis, mutual information, class imbalance detection, data leakage flagging |
| **LLM-Powered Planning** | Llama 3.3 70B generates encoding strategies, preprocessing steps, and model configs with reasoning |
| **Encoding Intelligence** | Explains WHY it chose one-hot vs target vs frequency vs label encoding for each feature |
| **Class Imbalance Handling** | Auto-detects imbalance ratio, applies `scale_pos_weight` / `class_weight='balanced'` |
| **SHAP Feature Importance** | TreeExplainer-based feature importance — shows what actually drives predictions |
| **Smart Hyperparameters** | Dataset-size-aware model configuration (small/medium/large datasets) |
| **Rate Limit Safe** | Single efficient LLM call, response caching, exponential backoff, rule-based fallback |
| **Interactive Dashboard** | Streamlit UI with real-time training, visualizations, and natural language chat |

---

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  CSV Upload │────▶│  Data Profiler   │────▶│  LLM Planner (Groq) │
└─────────────┘     │                  │     │                     │
                    │ • Correlations   │     │ • Encoding strategy │
                    │ • Mutual Info    │     │ • Preprocessing     │
                    │ • Imbalance      │     │ • Model config      │
                    │ • Leakage detect │     │ • Metric selection  │
                    └──────────────────┘     └──────────┬──────────┘
                                                       │
                    ┌──────────────────┐               ▼
                    │   Results + Chat │◀────┌─────────────────────┐
                    │                  │     │  Feature Engineer   │
                    │ • SHAP plots     │     │                     │
                    │ • Model compare  │     │ • Target encoding   │
                    │ • Encoding why   │     │ • One-hot encoding  │
                    │ • NL questions   │     │ • Frequency encoding│
                    └──────────────────┘     │ • Log transforms    │
                           ▲                 │ • Missing handling  │
                           │                 └──────────┬──────────┘
                    ┌──────┴───────────┐               │
                    │  Baseline Models │◀───────────────┘
                    │                  │
                    │ • LightGBM       │
                    │ • XGBoost        │
                    │ • Random Forest  │
                    │ • Stratified CV  │
                    └──────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- Groq API key (free at [console.groq.com](https://console.groq.com) — no credit card needed)

### Installation

```bash
git clone https://github.com/Shivansh-707/llm-powered-automl.git
cd llm-powered-automl

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API key
echo "GROQ_API_KEY=your_api_key_here" > .env
```

### Verify Setup

```bash
python test_setup.py
# Should print: "Setup complete!"
```

### Run the Application

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## How It Works

### 1. Upload & Profile
Upload any CSV dataset. The system performs deep statistical profiling:
- Per-feature skewness, cardinality, null percentages
- Pairwise correlation analysis (flags pairs > 0.5)
- Mutual information scoring (ranks features by predictive power)
- Class imbalance ratio calculation
- Data leakage detection (features > 0.95 correlated with target)

### 2. LLM Planning (Single Efficient Call)
The rich profile is sent to Llama 3.3 70B with an expert prompt containing:
- Explicit encoding rules (when to use one-hot vs target vs frequency)
- Missing value strategies based on percentage thresholds
- Class imbalance handling rules
- Metric selection logic
- Dataset-size-aware hyperparameter guidance

The LLM returns a structured JSON plan with reasoning for every decision.

### 3. Feature Engineering
The plan is executed with:
- **One-hot encoding**: For categoricals with ≤10 unique values
- **Target encoding**: For medium cardinality (10-50) with smoothing
- **Frequency encoding**: For high cardinality (>50) to prevent overfitting
- **Label encoding**: For ordinal features
- **Log/Log1p transforms**: For right-skewed numerics
- **Missing value indicators**: Binary flags + imputation for informative missingness

### 4. Model Training
Three baseline models trained with intelligent configuration:
- **LightGBM**: `scale_pos_weight` for imbalance, adaptive `n_estimators`
- **XGBoost**: Same imbalance handling, early stopping ready
- **Random Forest**: `class_weight='balanced'`, adaptive `min_samples_leaf`
- **Stratified K-Fold CV** for classification (preserves class distribution)

### 5. SHAP Explainability
After training, SHAP TreeExplainer computes feature importance:
- Shows which features actually drive predictions (not just split frequency)
- Handles binary, multi-class, and regression
- Samples large datasets for speed (configurable)

### 6. Chat Interface
Ask questions about your results in natural language:
- "Why did the model choose these features?"
- "What's the imbalance ratio?"
- "Which encoding was used for column X and why?"

---

## Example Results

### Synthetic Credit Default Dataset (2,000 rows, 9 features, 5.35:1 imbalance)

**LLM Reasoning:**
> Given the medium-sized dataset with class imbalance (5.35:1), the strategy focuses on F1-macro as the evaluation metric, applies scale_pos_weight to handle imbalance, and uses target encoding for high-cardinality merchant_id to prevent dimensionality explosion.

**Encoding Decisions:**
| Feature | Encoding | Reason |
|---------|----------|--------|
| merchant_id (200 unique) | Target | High cardinality — captures target relationship |
| loan_purpose (25 unique) | Target | Medium cardinality — target encoding preserves info |
| education (4 unique) | One-hot | Low cardinality — preserves all information |
| city (7 unique) | One-hot | Low cardinality — no ordinal relationship |

**Model Performance (F1-macro, 5-fold Stratified CV):**
| Model | Score | Std |
|-------|-------|-----|
| LightGBM | 0.8000 | ±0.0055 |
| Random Forest | 0.7975 | ±0.0091 |
| XGBoost | 0.7605 | ±0.0196 |

**SHAP Top Features:**
1. merchant_id: 0.7952
2. loan_amount: 0.3999
3. income: 0.3977
4. credit_score: 0.3799
5. employment_years: 0.3765

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Llama 3.3 70B via Groq API |
| ML Models | LightGBM, XGBoost, scikit-learn |
| Explainability | SHAP (TreeExplainer) |
| Feature Engineering | category-encoders, scipy, sklearn |
| Web Framework | Streamlit |
| Visualization | Plotly |
| Data Processing | Pandas, NumPy |

---

## Rate Limit Strategy

Groq free tier gives 30 RPM / 1,000 RPD / 12K TPM for Llama 3.3 70B. This system stays safe by:

1. **Single consolidated prompt** — One LLM call per dataset (not multi-turn)
2. **Response caching** — Same dataset profile → cached plan (no duplicate calls)
3. **Compact profile** — Sends only decision-relevant data to minimize tokens
4. **Exponential backoff** — 10s → 20s → 40s retry on 429 errors
5. **Rule-based fallback** — If LLM is unavailable, applies sensible defaults

---

## Project Structure

```
llm-powered-automl/
├── app.py                    # Streamlit dashboard (entry point)
├── config.py                 # Configuration (API keys, constants)
├── requirements.txt          # Python dependencies
├── test_setup.py             # API connectivity test
├── test_pipeline.py          # Full integration test
├── .env                      # API key (not committed)
├── .gitignore
├── README.md
└── src/
    ├── __init__.py
    ├── data_profiler.py      # Statistical profiling engine
    ├── llm_planner.py        # Groq LLM integration + fallback
    ├── feature_engineering.py # All encoding/transform operations
    ├── modeling.py           # Model training + SHAP
    ├── tools.py              # Chat context utilities
    └── utils.py              # Experiment memory
```

---

## Configuration

All settings in `config.py`:

```python
MODEL_NAME = "llama-3.3-70b-versatile"  # Groq model
CV_FOLDS = 5                            # Cross-validation folds
MAX_CARDINALITY_ONEHOT = 10             # One-hot threshold
HIGH_CARDINALITY_THRESHOLD = 50         # Target encoding threshold
SHAP_SAMPLE_SIZE = 500                  # SHAP computation sample
SHAP_TOP_FEATURES = 20                  # Top features to display
```

---

## Future Enhancements

- [ ] Hyperparameter optimization with Optuna
- [ ] Multi-class SHAP summary plots (beeswarm)
- [ ] Experiment comparison across datasets
- [ ] Model export (pickle/ONNX)
- [ ] ROC curves and confusion matrices
- [ ] Support for Excel and Parquet files
- [ ] Automated PDF/HTML reports
- [ ] Time-series aware CV splitting

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Author

**Shivansh Jha**
- Final Year CSE Student, India
- Lok Jagruti University
- Kaggle Enthusiast
- GitHub: [@Shivansh-707](https://github.com/Shivansh-707)

---

## License

This project is licensed under the MIT License.

---

Made with Llama 3.3 70B on Groq • LightGBM • XGBoost • SHAP
