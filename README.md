# 🤖 LLM-Powered AutoML System

An intelligent automated machine learning system that uses **Google Gemini AI** to analyze datasets, generate feature engineering strategies, and train baseline models with natural language reasoning.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30+-FF4B4B.svg)](https://streamlit.io)
[![Gemini AI](https://img.shields.io/badge/Gemini-2.5%20Flash-4285F4.svg)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **What makes this different?** Traditional AutoML uses hardcoded rules. This system uses LLM reasoning to make intelligent, context-aware decisions about feature engineering and modeling strategies.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **LLM-Powered Analysis** | Gemini AI analyzes dataset characteristics and generates reasoning-based strategies |
| 🔧 **Smart Feature Engineering** | Automatic encoding selection (target, one-hot, label) based on cardinality and data distribution |
| 🤖 **Multi-Model Training** | Trains LightGBM, XGBoost, and Random Forest with 5-fold cross-validation |
| 📊 **Interactive Dashboard** | Real-time Streamlit interface for dataset upload, training, and visualization |
| 💬 **Natural Language Chat** | Ask questions about your model in plain English and get intelligent answers |
| 📈 **Advanced Visualizations** | Model comparison charts, feature importance rankings, and performance metrics |
| 💾 **Experiment Memory** | Stores past experiments to learn from similar datasets |

---

## 🎯 Why This Project Matters

### The Problem
Most AutoML systems use **hardcoded heuristics**:
```python
if cardinality > 50:
    use_target_encoding()
else:
    use_onehot_encoding()
The Solution
This system uses LLM reasoning for context-aware decisions:

text
LLM: "The 'city' column has 250 unique values and a 0.45 correlation 
with the target. Target encoding is preferred because it preserves 
the relationship while avoiding dimensionality explosion."
The LLM provides interpretable reasoning for every decision, making the system educational and trustworthy.

🚀 Quick Start
Prerequisites
Python 3.9 or higher

Google Gemini API key (Get one free here - no credit card required)

Installation
bash
# 1. Clone the repository
git clone https://github.com/Shivansh-707/llm-powered-automl.git
cd llm-powered-automl

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your API key
echo "GEMINI_API_KEY=your_api_key_here" > .env
⚠️ Important: Replace your_api_key_here with your actual Gemini API key

Run the Application
bash
streamlit run app.py
Open http://localhost:8501 in your browser and you're ready to go! 🎉

📖 How It Works
Architecture Overview
text
┌─────────────────┐
│  Upload CSV     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Data Profiler          │
│  -  Analyze dtypes       │
│  -  Detect cardinality   │
│  -  Calculate skewness   │
│  -  Identify missing     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  LLM Planner (Gemini)   │
│  -  Generate reasoning   │
│  -  Plan preprocessing   │
│  -  Design features      │
│  -  Select models        │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Feature Engineer       │
│  -  Apply encodings      │
│  -  Handle missing       │
│  -  Transform features   │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Baseline Modeler       │
│  -  Train LightGBM       │
│  -  Train XGBoost        │
│  -  Train Random Forest  │
│  -  Cross-validation     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Results & Visualization│
│  -  Model comparison     │
│  -  Feature importance   │
│  -  Chat interface       │
└─────────────────────────┘
Workflow Example
1. Dataset Upload

text
Heart Disease Dataset: 630,000 rows × 15 columns
Target: Heart Disease (binary classification)
2. LLM Analysis

text
Gemini AI identifies:
- ST depression has skewness 1.328 → Apply log transform
- City has 250 unique values → Use target encoding
- Sex has 2 values → Use one-hot encoding
- ID column → Drop (no predictive value)
3. Execution

text
✓ Applied 9 transformations
✓ Trained 3 models with 5-fold CV
✓ Best: LightGBM with 95.46% ROC-AUC
4. Interactive Analysis

text
User: "Why did LightGBM perform better?"
AI: "LightGBM achieved 95.46% ROC-AUC vs Random Forest's 94.74% 
     because it handles high-dimensional sparse features more 
     efficiently after one-hot encoding..."
🏆 Example Results
Kaggle Playground series season 6 episode 2 dataset for Heart Disease
Heart Disease Classification Dataset
Dataset Stats:

630,000 rows

13 predictive features

Binary classification (Presence/Absence)

No missing values

Balanced classes (55%/45%)

Model Performance:

Model	ROC-AUC	Std Dev	Training Time
🥇 LightGBM	0.9546	±0.0012	23.4s
🥈 XGBoost	0.9523	±0.0015	31.7s
🥉 Random Forest	0.9474	±0.0018	45.2s
Top 5 Most Important Features:

Thallium (0.24)

Number of vessels fluro (0.18)

Chest pain type (0.16)

ST depression (0.14)

Max HR (0.12)

LLM Reasoning Excerpt:

"The dataset presents a binary classification problem with excellent data quality (no missing values). ST depression exhibits significant skewness (1.328), so a log transformation will normalize its distribution. Tree-based ensemble models are ideal for this tabular data with mixed feature types."

🛠️ Tech Stack
Core Technologies
Component	Technology	Purpose
LLM	Google Gemini 2.5 Flash	Dataset analysis and reasoning
ML Models	LightGBM, XGBoost, scikit-learn	Baseline model training
Feature Engineering	category-encoders, scipy	Encoding and transformations
Web Framework	Streamlit	Interactive dashboard
Visualization	Plotly	Charts and graphs
Data Processing	Pandas, NumPy	Data manipulation
Why These Choices?
Gemini 2.5 Flash: Free tier, 1M context window, fast inference

LightGBM: Efficient on large datasets, handles categorical features natively

Streamlit: Rapid prototyping, automatic reactivity, clean UI

Plotly: Interactive charts, professional visualizations

📂 Project Structure
text
llm-powered-automl/
├── src/
│   ├── __init__.py
│   ├── data_profiler.py      # Dataset analysis & statistics
│   ├── llm_planner.py         # Gemini AI integration & prompts
│   ├── feature_engineering.py # Encoding & transformation logic
│   ├── modeling.py            # Model training & evaluation
│   ├── tools.py               # Chat interface tools
│   └── utils.py               # Helper functions & memory
├── app.py                     # Streamlit dashboard (main entry)
├── config.py                  # Configuration & API keys
├── requirements.txt           # Python dependencies
├── .env                       # API keys (not committed)
├── .gitignore                 # Files to ignore in git
└── README.md                  # This file
Key Design Principles:

✅ Modular: Each component is independent and testable

✅ Extensible: Easy to add new encodings or models

✅ Maintainable: Clean separation of concerns

✅ Type-safe: Defensive programming with .get() methods

🎓 What I Learned Building This
Technical Skills
Prompt Engineering

Crafting prompts for structured JSON output

Handling LLM response parsing and error recovery

Providing context for better reasoning

LLM Integration

Google Gemini API usage and best practices

Managing API rate limits (15 RPM on free tier)

Implementing fallback strategies for JSON extraction

AutoML Pipeline Design

Dataset profiling with statistical analysis

Dynamic strategy selection based on data characteristics

Cross-validation and metric selection

Feature Engineering at Scale

Target encoding for high-cardinality features

Log transformations for skewed distributions

Handling mixed data types (numeric + categorical)

Full-Stack ML Development

Streamlit for reactive UIs

Session state management

Real-time model training feedback

Challenges Overcome
Challenge	Solution
LLM returning invalid JSON	Regex fallback + markdown parsing
Large dataset memory issues	Efficient pandas dtypes + chunking
API rate limits	Caching profiling results
Mixed data types	Type detection + automatic handling
🔮 Future Enhancements
Planned Features
 Hyperparameter Optimization: Integrate Optuna for automated tuning

 Multi-Agent Architecture: Separate agents for feature engineering and modeling

 Experiment Tracking: Compare current run with past experiments

 Model Export: Save trained models as pickle/ONNX for deployment

 Advanced Metrics: ROC curves, confusion matrices, SHAP values

 File Format Support: Excel, Parquet, JSON input

 Report Generation: Automated PDF/HTML reports with findings

 Deployment Ready: Docker containerization + API endpoint

