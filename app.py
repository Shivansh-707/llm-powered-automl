import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.data_profiler import profile_dataset
from src.llm_planner import generate_automl_plan, chat_with_context
from src.feature_engineering import FeatureEngineer
from src.modeling import BaselineModeler
from src.tools import AutoMLTools
from src.utils import save_experiment, get_dataset_signature
import json

st.set_page_config(page_title="LLM-Powered AutoML", layout="wide", page_icon="🤖")

# Title
st.title("🤖 LLM-Powered AutoML System")
st.markdown("*Expert-level feature engineering and modeling powered by Llama 3.3 70B on Groq*")

# Sidebar
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv'])

# Initialize session state
for key in ['plan', 'results', 'df_processed', 'modeler', 'feature_engineer',
            'chat_history', 'profile', 'shap_importance']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'chat_history' else []

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    st.sidebar.success(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Target column selection
    target_col = st.sidebar.selectbox("Select Target Column", df.columns)

    # Show dataset preview
    with st.expander("📊 Dataset Preview", expanded=False):
        st.dataframe(df.head(10))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Features", df.shape[1] - 1)
        col3.metric("Missing Cells", f"{df.isnull().sum().sum():,}")
        col4.metric("Duplicates", f"{df.duplicated().sum():,}")

    # Main action button
    if st.sidebar.button("🚀 Generate AutoML Plan", type="primary"):
        with st.spinner("🔬 Profiling dataset (correlations, mutual info, distributions)..."):
            profile = profile_dataset(df, target_col)
            st.session_state.profile = profile

        with st.spinner("🧠 LLM is analyzing your dataset..."):
            plan = generate_automl_plan(profile)

            if plan:
                st.session_state.plan = plan
                st.success("✅ AutoML plan generated!")
            else:
                st.error("❌ Failed to generate plan. Check your API key or try again.")

    # Display plan and execute
    if st.session_state.plan:
        st.header("📋 Generated AutoML Plan")

        # Show reasoning
        with st.expander("🧠 LLM Reasoning", expanded=True):
            st.markdown(st.session_state.plan.get('reasoning', 'No reasoning provided'))

            # Show encoding rationale
            encoding_rationale = st.session_state.plan.get('encoding_rationale', '')
            if encoding_rationale:
                st.markdown("---")
                st.markdown(f"**Encoding Philosophy:** {encoding_rationale}")

        # Show plan details
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔧 Data Preprocessing")
            preprocessing = st.session_state.plan.get('data_preprocessing', [])
            if preprocessing:
                for step in preprocessing:
                    column = step.get('column', 'N/A')
                    strategy = step.get('strategy', step.get('action', 'N/A'))
                    reason = step.get('reason', '')
                    st.markdown(f"**{column}**: `{strategy}`")
                    if reason:
                        st.caption(reason)
            else:
                st.info("No preprocessing needed — clean dataset!")

        with col2:
            st.subheader("⚙️ Feature Engineering")
            feat_eng = st.session_state.plan.get('feature_engineering', [])
            if feat_eng:
                for step in feat_eng:
                    column = step.get('column', 'N/A')
                    action = step.get('action', 'N/A')
                    reason = step.get('reason', '')
                    st.markdown(f"**{column}**: `{action}`")
                    if reason:
                        st.caption(reason)
            else:
                st.info("No feature engineering needed")

        # Show model configuration
        with st.expander("🎛️ Model Configuration", expanded=False):
            model_params = st.session_state.plan.get('model_params', {})
            eval_info = st.session_state.plan.get('evaluation', {})

            if eval_info:
                st.markdown(f"**Metric:** `{eval_info.get('metric', 'accuracy')}` — {eval_info.get('reason', '')}")

            if model_params:
                for model_name, params in model_params.items():
                    st.markdown(f"**{model_name}:** {json.dumps(params, indent=1)}")

            challenges = st.session_state.plan.get('expected_challenges', [])
            if challenges:
                st.markdown("**Expected Challenges:**")
                for c in challenges:
                    st.markdown(f"- {c}")

        # Execute plan button
        if st.button("▶️ Execute Plan & Train Models", type="primary"):
            with st.spinner("🔄 Applying transformations..."):
                try:
                    # Apply feature engineering
                    feature_engineer = FeatureEngineer()
                    df_processed = feature_engineer.apply_plan(
                        df.copy(),
                        target_col,
                        st.session_state.plan
                    )

                    # Prepare data for modeling
                    X = df_processed.drop(columns=[target_col])
                    y = df_processed[target_col]

                    # Handle any remaining non-numeric columns (safety net)
                    # Use intelligent encoding rather than blind category codes
                    remaining_obj = X.select_dtypes(include=['object']).columns
                    for col in remaining_obj:
                        cardinality = X[col].nunique()
                        if cardinality <= 10:
                            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True).astype(int)
                            X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
                        else:
                            # Frequency encoding for remaining high-cardinality
                            freq_map = X[col].value_counts(normalize=True).to_dict()
                            X[col] = X[col].map(freq_map).fillna(0)

                    # Handle any remaining NaN
                    X = X.fillna(0)

                    # Determine problem type and metric
                    problem_type = st.session_state.plan.get('target_info', {}).get('type', 'classification')
                    metric = st.session_state.plan.get('evaluation', {}).get('metric', 'accuracy')
                    model_params = st.session_state.plan.get('model_params', None)

                    # Get imbalance ratio
                    imbalance_ratio = 1.0
                    if st.session_state.profile:
                        imbalance_ratio = st.session_state.profile.get(
                            'class_balance', {}
                        ).get('imbalance_ratio', 1.0)

                    st.info(f"Training with metric: **{metric}** | Problem type: **{problem_type}** | Imbalance ratio: **{imbalance_ratio:.1f}:1**")

                except Exception as e:
                    st.error(f"❌ Error during feature engineering: {str(e)}")
                    st.stop()

            with st.spinner("🏋️ Training baseline models (LightGBM, XGBoost, Random Forest)..."):
                try:
                    modeler = BaselineModeler(problem_type=problem_type)
                    results = modeler.train_baseline(
                        X, y,
                        models_to_train=st.session_state.plan.get('baseline_models', ['lightgbm', 'xgboost', 'random_forest']),
                        metric=metric,
                        imbalance_ratio=imbalance_ratio,
                        model_params=model_params,
                    )
                except Exception as e:
                    st.error(f"❌ Error during model training: {str(e)}")
                    st.stop()

            with st.spinner("🔍 Computing SHAP feature importance..."):
                try:
                    feature_names = list(X.columns)
                    shap_importance = modeler.compute_shap_importance(X, feature_names)
                    st.session_state.shap_importance = shap_importance
                except Exception as e:
                    print(f"SHAP failed: {e}")
                    st.session_state.shap_importance = None

            # Save to session state
            st.session_state.results = results
            st.session_state.modeler = modeler
            st.session_state.feature_engineer = feature_engineer
            st.session_state.df_processed = df_processed

            # Save to memory
            signature = get_dataset_signature(df)
            save_experiment(signature, st.session_state.plan, results)

            st.success("✅ Training complete!")
            st.rerun()

    # Display results
    if st.session_state.results:
        st.header("📊 Model Results")

        # Model comparison
        results_data = []
        for model_name, metrics in st.session_state.results.items():
            if 'error' not in metrics:
                results_data.append({
                    'Model': model_name,
                    'Score': metrics['cv_mean'],
                    'Std': metrics['cv_std'],
                    'Metric': metrics['metric']
                })

        if results_data:
            results_df = pd.DataFrame(results_data)

            # Bar chart
            fig = px.bar(
                results_df,
                x='Model',
                y='Score',
                error_y='Std',
                title=f'Model Performance Comparison ({results_data[0]["Metric"]})',
                color='Score',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Best model highlight
            best_model = st.session_state.modeler.best_model
            best_score = abs(st.session_state.modeler.best_score)

            col1, col2, col3 = st.columns(3)
            col1.metric("🏆 Best Model", best_model)
            col2.metric("📈 Best Score", f"{best_score:.4f}")
            col3.metric("📊 Metric", st.session_state.results[best_model]['metric'])

            # SHAP Feature Importance
            st.subheader("🎯 Feature Importance (SHAP)")

            shap_imp = st.session_state.shap_importance
            if shap_imp and 'shap_importance' in shap_imp:
                importance_df = pd.DataFrame({
                    'features': shap_imp['features'],
                    'importance': shap_imp['shap_importance']
                })
                fig_imp = px.bar(
                    importance_df,
                    x='importance',
                    y='features',
                    orientation='h',
                    title=f'SHAP Feature Importance (from {shap_imp["model_used"]})',
                    color='importance',
                    color_continuous_scale='Reds',
                )
                fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_imp, use_container_width=True)

                st.caption("SHAP values show the average impact of each feature on model predictions. Higher = more important.")
            elif shap_imp and 'importances' in shap_imp:
                # Fallback to split-based importance
                importance_df = pd.DataFrame({
                    'features': shap_imp['features'],
                    'importance': shap_imp['importances']
                })
                fig_imp = px.bar(
                    importance_df,
                    x='importance',
                    y='features',
                    orientation='h',
                    title=f'Feature Importance — split-based fallback (from {shap_imp.get("model_used", "best model")})',
                )
                fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_imp, use_container_width=True)
                st.caption("Split-based importance (SHAP was unavailable). Shows how often each feature is used in tree splits.")
            else:
                st.info("Feature importance not available.")

            # Encoding decisions
            if st.session_state.feature_engineer:
                decisions = st.session_state.feature_engineer.encoding_decisions
                if decisions:
                    with st.expander("🏷️ Encoding Decisions Explained", expanded=False):
                        for d in decisions:
                            st.markdown(f"**{d['column']}** → `{d['encoding']}` — {d['reason']}")

        # Chat interface
        st.header("💬 Ask Questions About Your Model")

        user_question = st.text_input("Ask anything about your dataset, features, or results:")

        if user_question:
            with st.spinner("🤔 Thinking..."):
                # Build context
                context = {
                    "dataset_shape": list(st.session_state.df_processed.shape),
                    "target": target_col,
                    "best_model": st.session_state.modeler.best_model,
                    "best_score": float(abs(st.session_state.modeler.best_score)),
                    "results": {k: {kk: vv for kk, vv in v.items() if kk != 'cv_scores'}
                                for k, v in st.session_state.results.items()},
                    "transformations": st.session_state.feature_engineer.get_summary(),
                }

                if st.session_state.shap_importance:
                    context["top_features_shap"] = st.session_state.shap_importance.get('features', [])[:10]

                # Get LLM response
                response = chat_with_context(user_question, context)

                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": response
                })

        # Display chat history
        if st.session_state.chat_history:
            st.subheader("💭 Conversation History")
            for chat in reversed(st.session_state.chat_history[-5:]):
                with st.container():
                    st.markdown(f"**Q:** {chat['question']}")
                    st.markdown(f"**A:** {chat['answer']}")
                    st.divider()

else:
    # Welcome screen
    st.info("👈 Upload a CSV file from the sidebar to get started!")

    st.markdown("""
    ### 🚀 How it works:

    1. **Upload** your dataset (CSV format)
    2. **Select** your target column
    3. **Generate** an expert AutoML plan using Llama 3.3 70B
    4. **Execute** the plan — smart encoding, transformations, and model training
    5. **Analyze** results with SHAP feature importance and chat

    ### ✨ What makes this expert-level:

    - 🧠 **Intelligent encoding** — Explains WHY it chose one-hot vs target vs frequency encoding
    - ⚖️ **Class imbalance handling** — Auto-detects and adjusts model weights
    - 🔍 **SHAP importance** — Shows which features actually drive predictions
    - 📊 **Smart profiling** — Correlations, mutual information, leakage detection
    - 🛡️ **Rate limit safe** — Single efficient LLM call with caching and fallback
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Powered by Groq (Llama 3.3 70B) • LightGBM • XGBoost • SHAP")
