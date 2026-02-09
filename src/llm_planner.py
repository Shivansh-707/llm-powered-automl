import google.generativeai as genai
import json
import re
from config import GEMINI_API_KEY, MODEL_NAME

genai.configure(api_key=GEMINI_API_KEY)

def generate_automl_plan(profile):
    """
    Use Gemini to analyze dataset profile and generate
    a comprehensive AutoML plan with reasoning.
    """
    
    prompt = f"""You are an expert AutoML system. Analyze this dataset profile and create a detailed machine learning plan.

Dataset Profile:
{json.dumps(profile, indent=2)}

Generate a JSON plan with the following structure:
{{
  "reasoning": "Detailed analysis of the dataset characteristics and why you chose these strategies",
  "data_preprocessing": [
    {{"action": "handle_missing", "column": "column_name", "strategy": "mean/median/mode/drop", "reason": "why"}},
    ...
  ],
  "feature_engineering": [
    {{"action": "target_encode/onehot_encode/label_encode/log_transform/polynomial", "column": "column_name", "params": {{}}, "reason": "why"}},
    ...
  ],
  "feature_selection": {{
    "apply": true/false,
    "method": "feature_importance/correlation",
    "reason": "why or why not"
  }},
  "baseline_models": ["lightgbm", "xgboost", "random_forest"],
  "evaluation": {{
    "strategy": "cross_validation",
    "folds": 5,
    "metric": "accuracy/roc_auc/rmse/mae based on problem type"
  }},
  "expected_challenges": ["potential issues to watch for"]
}}

Key considerations:
1. For high cardinality categorical features (>50 unique values), prefer target encoding
2. For low cardinality (<10 unique values), use one-hot encoding
3. Handle missing values appropriately based on percentage and feature type
4. Consider log transforms for highly skewed numeric features (skewness > 1 or < -1)
5. For classification, choose appropriate metric (accuracy, roc_auc, f1)
6. For regression, choose appropriate metric (rmse, mae, r2)

Return ONLY valid JSON, no markdown formatting or extra text."""

    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    
    response_text = response.text.strip()
    
    # Clean up markdown code blocks if present
    if "```json" in response_text:
        # Extract between ```json and next ```
        start = response_text.find("```json") + 7
        end = response_text.find("```", start)
        if end != -1:
            response_text = response_text[start:end].strip()
    elif "```" in response_text:
        # Extract between first ``` and second ```
        parts = response_text.split("```")
        if len(parts) >= 3:
            response_text = parts[1].strip()
    
    # Try to parse JSON
    try:
        plan = json.loads(response_text)
        return plan
    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM response: {e}")
        print(f"Response text: {response_text[:500]}...")
        
        # Try to find JSON in the response with regex
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                plan = json.loads(json_match.group())
                print("✓ Successfully extracted JSON with regex")
                return plan
            except:
                print("✗ Regex extraction also failed")
        
        return None


def chat_with_context(user_message, context):
    """
    Simple chat interface that uses context to answer questions.
    """
    
    prompt = f"""You are an AutoML assistant helping a user understand their ML pipeline.

Current Context:
{json.dumps(context, indent=2, default=str)}

User Question: {user_message}

Provide a helpful, concise answer based on the context. If you need to suggest actions, be specific."""

    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    
    return response.text
