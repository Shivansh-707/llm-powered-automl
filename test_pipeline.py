"""
Full integration test of the AutoML pipeline.
Creates a synthetic dataset and runs the entire flow:
  profile → LLM plan → feature engineering → model training → SHAP
"""
import pandas as pd
import numpy as np
from src.data_profiler import profile_dataset
from src.llm_planner import generate_automl_plan
from src.feature_engineering import FeatureEngineer
from src.modeling import BaselineModeler

np.random.seed(42)

# --- Create a realistic synthetic dataset ---
n = 2000
df = pd.DataFrame({
    'age': np.random.randint(18, 80, n),
    'income': np.random.exponential(50000, n),  # Right-skewed
    'credit_score': np.random.normal(650, 100, n).astype(int),
    'employment_years': np.random.exponential(5, n),  # Skewed
    'loan_amount': np.random.exponential(20000, n),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix', 'Dallas', 'Austin'], n),
    'education': np.random.choice(['high_school', 'bachelors', 'masters', 'phd'], n),
    'loan_purpose': np.random.choice([f'purpose_{i}' for i in range(25)], n),  # Medium cardinality
    'merchant_id': np.random.choice([f'merchant_{i}' for i in range(200)], n),  # High cardinality
    'default': np.random.choice([0, 1], n, p=[0.85, 0.15]),  # Imbalanced target
})

# Add some missing values
df.loc[np.random.choice(n, 50, replace=False), 'income'] = np.nan
df.loc[np.random.choice(n, 30, replace=False), 'credit_score'] = np.nan
df.loc[np.random.choice(n, 100, replace=False), 'city'] = np.nan

target_col = 'default'

print("=" * 60)
print("STEP 1: Data Profiling")
print("=" * 60)
profile = profile_dataset(df, target_col)
print(f"  Rows: {profile['basic_info']['n_rows']}")
print(f"  Features: {profile['basic_info']['n_features']}")
print(f"  Target type: {profile['target_info']['type']}")
print(f"  Class imbalance: {profile['class_balance']}")
print(f"  Missing %: {profile['data_quality']['total_missing_pct']}%")
print(f"  High correlations: {len(profile['correlations'].get('high_pairs', []))}")
print(f"  Top predictive features: {[f['feature'] for f in profile['correlations'].get('mutual_info_top10', [])[:5]]}")

print("\n" + "=" * 60)
print("STEP 2: LLM Plan Generation (Groq - Llama 3.3 70B)")
print("=" * 60)
plan = generate_automl_plan(profile)

if plan:
    print(f"  ✓ Plan generated successfully")
    print(f"  Reasoning: {plan.get('reasoning', 'N/A')[:150]}...")
    print(f"  Encoding rationale: {plan.get('encoding_rationale', 'N/A')[:150]}")
    print(f"  Preprocessing steps: {len(plan.get('data_preprocessing', []))}")
    print(f"  Feature engineering steps: {len(plan.get('feature_engineering', []))}")
    print(f"  Metric: {plan.get('evaluation', {}).get('metric', 'N/A')}")
    print(f"  Models: {plan.get('baseline_models', [])}")
else:
    print("  ✗ Plan generation failed, using fallback")
    from src.llm_planner import _fallback_plan
    plan = _fallback_plan(profile)

print("\n" + "=" * 60)
print("STEP 3: Feature Engineering")
print("=" * 60)
fe = FeatureEngineer()
df_processed = fe.apply_plan(df.copy(), target_col, plan)

# Handle remaining object columns
X = df_processed.drop(columns=[target_col])
y = df_processed[target_col]
for col in X.select_dtypes(include=['object']).columns:
    X[col] = pd.Categorical(X[col]).codes
X = X.fillna(0)

print(f"  Original shape: {df.shape}")
print(f"  Processed shape: {X.shape}")
print(f"  Transformations applied: {len(fe.transformations_applied)}")
for t in fe.transformations_applied:
    print(f"    • {t}")
print(f"\n  Encoding decisions:")
for d in fe.encoding_decisions:
    print(f"    • {d['column']}: {d['encoding']} — {d['reason']}")

print("\n" + "=" * 60)
print("STEP 4: Model Training")
print("=" * 60)
problem_type = plan.get('target_info', {}).get('type', 'classification')
metric = plan.get('evaluation', {}).get('metric', 'roc_auc')
imbalance_ratio = profile.get('class_balance', {}).get('imbalance_ratio', 1.0)
model_params = plan.get('model_params', None)

modeler = BaselineModeler(problem_type=problem_type)
results = modeler.train_baseline(
    X, y,
    models_to_train=plan.get('baseline_models', ['lightgbm', 'xgboost', 'random_forest']),
    metric=metric,
    imbalance_ratio=imbalance_ratio,
    model_params=model_params,
)

print(f"\n  Best model: {modeler.best_model}")
print(f"  Best score: {abs(modeler.best_score):.4f}")

print("\n" + "=" * 60)
print("STEP 5: SHAP Feature Importance")
print("=" * 60)
feature_names = list(X.columns)
shap_result = modeler.compute_shap_importance(X, feature_names)

if shap_result:
    print(f"  Method: {shap_result['method']}")
    print(f"  Model: {shap_result['model_used']}")
    print(f"  Top features:")
    values_key = 'shap_importance' if 'shap_importance' in shap_result else 'importances'
    for feat, imp in zip(shap_result['features'][:10], shap_result[values_key][:10]):
        print(f"    • {feat}: {imp:.4f}")
else:
    print("  ✗ SHAP computation failed")

print("\n" + "=" * 60)
print("✅ PIPELINE TEST COMPLETE")
print("=" * 60)
