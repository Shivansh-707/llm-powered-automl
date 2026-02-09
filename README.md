# LLM-Powered AutoML System ( my kaggle journey inspiration ) 

An intelligent automated machine learning system that uses Google Gemini AI to analyze datasets, generate feature engineering strategies, and train baseline models with natural language reasoning.

## What Makes This Different?

Traditional AutoML uses hardcoded rules. This system uses LLM reasoning to make intelligent, context-aware decisions about feature engineering and modeling strategies.

The LLM provides interpretable reasoning for every decision, making the system educational and trustworthy.

## Key Features

- **LLM-Powered Analysis**: Gemini AI analyzes dataset characteristics and generates reasoning-based strategies
- **Smart Feature Engineering**: Automatic encoding selection (target, one-hot, label) based on cardinality and data distribution
- **Multi-Model Training**: Trains LightGBM, XGBoost, and Random Forest with 5-fold cross-validation
- **Interactive Dashboard**: Real-time Streamlit interface for dataset upload, training, and visualization
- **Natural Language Chat**: Ask questions about your model in plain English and get intelligent answers
- **Advanced Visualizations**: Model comparison charts, feature importance rankings, and performance metrics
- **Experiment Memory**: Stores past experiments to learn from similar datasets

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Google Gemini API key (get one free at https://aistudio.google.com/apikey - no credit card required)

### Installation

git clone https://github.com/Shivansh-707/llm-powered-automl.git
cd llm-powered-automl
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_api_key_here" > .env

text

Replace `your_api_key_here` with your actual Gemini API key.

### Run the Application

streamlit run app.py

text

Open http://localhost:8501 in your browser.

## How It Works

1. **Upload Dataset**: Upload a CSV file through the web interface
2. **Data Profiling**: System analyzes dtypes, cardinality, skewness, and missing values
3. **LLM Planning**: Gemini AI generates reasoning and creates a feature engineering plan
4. **Feature Engineering**: Applies encodings and transformations based on the plan
5. **Model Training**: Trains multiple baseline models with cross-validation
6. **Results**: View model comparison, feature importance, and ask questions via chat

## Example Results

### Heart Disease Classification Dataset

Dataset: 630,000 rows, 13 predictive features, binary classification

**Model Performance:**

- LightGBM: 95.46% ROC-AUC (±0.0012)
- XGBoost: 95.23% ROC-AUC (±0.0015)
- Random Forest: 94.74% ROC-AUC (±0.0018)

**LLM Reasoning:**

The dataset presents a binary classification problem with excellent data quality. ST depression exhibits significant skewness, so a log transformation will normalize its distribution. Tree-based ensemble models are ideal for this tabular data with mixed feature types.

## Tech Stack

- **LLM**: Google Gemini 2.5 Flash
- **ML Models**: LightGBM, XGBoost, scikit-learn
- **Feature Engineering**: category-encoders, scipy
- **Web Framework**: Streamlit
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy

## Project Structure

llm-powered-automl/
├── src/
│ ├── data_profiler.py # Dataset analysis
│ ├── llm_planner.py # Gemini AI integration
│ ├── feature_engineering.py # Encoding & transformations
│ ├── modeling.py # Model training
│ ├── tools.py # Chat interface
│ └── utils.py # Helper functions
├── app.py # Streamlit dashboard
├── config.py # Configuration
├── requirements.txt
└── README.md

text

## What I Learned

- Prompt engineering for structured JSON output from LLMs
- Google Gemini API integration and error handling
- AutoML pipeline design with modular architecture
- Feature engineering strategies for different data types
- Building interactive ML applications with Streamlit
- Cross-validation and metric selection for different problem types

## Future Enhancements

- Hyperparameter optimization with Optuna
- Multi-agent architecture for separate FE and modeling agents
- Experiment tracking and comparison
- Model export (pickle/ONNX)
- ROC curves and confusion matrices
- Support for Excel and Parquet files
- Automated PDF/HTML reports

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

**Shivansh Jha**

- Final Year CSE Student, India
- Data Science Intern @ PetPooja
- Kaggle Enthusiast
- GitHub: [@Shivansh-707](https://github.com/Shivansh-707)

## Acknowledgments

- Google Gemini AI for free LLM API access
- Streamlit for the web framework
- LightGBM/XGBoost teams for ML libraries
- Kaggle community for datasets
- Open-source ML community

## License

This project is licensed under the MIT License.

---

Made with ❤️ and Gemini AI ( and by my efforts as well ) 

⭐ If you find this project helpful, please consider giving it a star!
