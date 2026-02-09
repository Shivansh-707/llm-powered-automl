import streamlit as st
import pandas as pd
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
st.markdown("*Upload your dataset and let AI handle the feature engineering and modeling*")

# Sidebar
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv'])

# Initialize session state
if 'plan' not in st.session_state:
    st.session_state.plan = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'modeler' not in st.session_state:
    st.session_state.modeler = None
if 'feature_engineer' not in st.session_state:
    st.session_state.feature_engineer = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    
    st.sidebar.success(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Target column selection
    target_col = st.sidebar.selectbox("Select Target Column", df.columns)
    
    # Show dataset preview
    with st.expander("📊 Dataset Preview", expanded=False):
        st.dataframe(df.head(10))
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Features", df.shape[1])
        col3.metric("Missing Cells", df.isnull().sum().sum())
    
    # Main action button
    if st.sidebar.button("🚀 Generate AutoML Plan", type="primary"):
        with st.spinner("🧠 LLM is analyzing your dataset..."):
            # Profile dataset
            profile = profile_dataset(df, target_col)
            
            # Generate plan using LLM
            plan = generate_automl_plan(profile)
            
            if plan:
                st.session_state.plan = plan
                st.success("✅ AutoML plan generated!")
            else:
                st.error("❌ Failed to generate plan. Check your API key.")
    
    # Display plan and execute
    if st.session_state.plan:
        st.header("📋 Generated AutoML Plan")
        
        # Show reasoning
        with st.expander("🧠 LLM Reasoning", expanded=True):
            st.markdown(st.session_state.plan.get('reasoning', 'No reasoning provided'))
        
        # Show plan details
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 Data Preprocessing")
            preprocessing = st.session_state.plan.get('data_preprocessing', [])
            if preprocessing:
                for step in preprocessing:
                    column = step.get('column', 'N/A')
                    strategy = step.get('strategy', step.get('action', 'N/A'))
                    reason = step.get('reason', 'No reason provided')
                    st.markdown(f"**{column}**: {strategy}")
                    st.caption(reason)
            else:
                st.info("No preprocessing needed")
        
        with col2:
            st.subheader("⚙️ Feature Engineering")
            feat_eng = st.session_state.plan.get('feature_engineering', [])
            if feat_eng:
                for step in feat_eng:
                    column = step.get('column', 'N/A')
                    action = step.get('action', 'N/A')
                    reason = step.get('reason', 'No reason provided')
                    st.markdown(f"**{column}**: {action}")
                    st.caption(reason)
            else:
                st.info("No feature engineering needed")
        
        # Execute plan button
        if st.button("▶️ Execute Plan & Train Models", type="primary"):
            with st.spinner("🔄 Applying transformations and training models..."):
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
                    
                    # Handle any remaining non-numeric columns
                    for col in X.select_dtypes(include=['object']).columns:
                        X[col] = pd.Categorical(X[col]).codes
                    
                    # Determine problem type
                    problem_type = st.session_state.plan['target_info']['type'] if 'target_info' in st.session_state.plan else 'classification'
                    metric = st.session_state.plan.get('evaluation', {}).get('metric', 'accuracy')
                    
                    # Train models
                    modeler = BaselineModeler(problem_type=problem_type)
                    results = modeler.train_baseline(
                        X, y,
                        models_to_train=st.session_state.plan.get('baseline_models', ['lightgbm', 'xgboost']),
                        metric=metric
                    )
                    
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
                    
                except Exception as e:
                    st.error(f"❌ Error during execution: {str(e)}")
    
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
                title='Model Performance Comparison',
                color='Score',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Best model highlight
            best_model = st.session_state.modeler.best_model
            best_score = st.session_state.modeler.best_score
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🏆 Best Model", best_model)
            col2.metric("📈 Best Score", f"{best_score:.4f}")
            col3.metric("📊 Metric", st.session_state.results[best_model]['metric'])
            
            # Feature importance
            st.subheader("🎯 Feature Importance")
            feature_names = [col for col in st.session_state.df_processed.columns if col != target_col]
            importance = st.session_state.modeler.get_feature_importance(feature_names, top_n=15)
            
            if importance:
                importance_df = pd.DataFrame(importance)
                fig_imp = px.bar(
                    importance_df,
                    x='importances',
                    y='features',
                    orientation='h',
                    title='Top 15 Most Important Features'
                )
                st.plotly_chart(fig_imp, use_container_width=True)
        
        # Chat interface
        st.header("💬 Ask Questions About Your Model")
        
        user_question = st.text_input("Ask anything about your dataset or results:")
        
        if user_question:
            with st.spinner("🤔 Thinking..."):
                tools = AutoMLTools(
                    st.session_state.modeler,
                    st.session_state.feature_engineer,
                    st.session_state.df_processed,
                    target_col
                )
                
                # Build context
                context = {
                    "dataset_shape": st.session_state.df_processed.shape,
                    "target": target_col,
                    "best_model": st.session_state.modeler.best_model,
                    "best_score": st.session_state.modeler.best_score,
                    "results": st.session_state.results,
                    "transformations": st.session_state.feature_engineer.get_summary()
                }
                
                # Get LLM response
                response = chat_with_context(user_question, context)
                
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": response
                })
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("💭 Conversation History")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
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
    3. **Generate** AutoML plan using LLM reasoning
    4. **Execute** the plan and train baseline models
    5. **Analyze** results and chat with your AI assistant
    
    ### ✨ Features:
    
    - 🧠 **LLM-powered analysis** - Gemini AI analyzes your data
    - 🔧 **Smart feature engineering** - Automatic encoding strategies
    - 🤖 **Baseline models** - LightGBM, XGBoost, Random Forest
    - 📊 **Interactive visualizations** - Model comparison & feature importance
    - 💬 **Chat interface** - Ask questions about your results
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit & Gemini AI & shivansh JHA hehehe")