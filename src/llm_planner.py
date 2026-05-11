import json
import re
import time
import hashlib
from groq import Groq
from config import GROQ_API_KEY, MODEL_NAME

# Lazy client initialization — avoids crash if key isn't set at import time
_client = None


def _get_client():
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to .env (local) or Streamlit secrets (cloud)."
            )
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


# Simple in-memory cache to avoid duplicate calls
_plan_cache = {}


def _call_groq(messages, temperature=0.3, max_tokens=4000, retries=3):
    """
    Call Groq API with exponential backoff for rate limits.
    Keeps us safe on the free tier.
    """
    client = _get_client()
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str.lower():
                wait_time = (2 ** attempt) * 10  # 10s, 20s, 40s
                print(f"⏳ Rate limited. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"❌ Groq API error: {error_str}")
                if attempt < retries - 1:
                    time.sleep(2)
                else:
                    return None
    return None


def _parse_json_response(text):
    """Robustly extract JSON from LLM response."""
    if text is None:
        return None

    # Strip markdown code blocks
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try regex extraction
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


def generate_automl_plan(profile):
    """
    Generate an expert AutoML plan using Groq (Llama 3.3 70B).
    Single, well-crafted prompt to stay within rate limits.
    Uses caching to avoid duplicate calls for the same dataset.
    """
    # Cache key based on profile hash
    profile_hash = hashlib.md5(json.dumps(profile, sort_keys=True, default=str).encode()).hexdigest()
    if profile_hash in _plan_cache:
        print("✓ Using cached plan")
        return _plan_cache[profile_hash]

    # Compact the profile to save tokens
    compact_profile = _compact_profile(profile)

    prompt = f"""You are a senior ML engineer and Kaggle grandmaster. Analyze this dataset and produce an expert-level AutoML plan.

DATASET PROFILE:
{json.dumps(compact_profile, indent=1)}

RULES YOU MUST FOLLOW:
1. ENCODING SELECTION (be explicit about WHY):
   - One-hot: ONLY for categorical features with ≤10 unique values AND no ordinal relationship
   - Target encoding: For categorical features with >10 unique values (prevents dimensionality explosion)
   - Label encoding: ONLY for ordinal categoricals (where order matters, e.g., low/medium/high)
   - Frequency encoding: For high-cardinality categoricals where target encoding might overfit (small datasets)

2. MISSING VALUES:
   - Numeric with <5% missing: median (robust to outliers)
   - Numeric with 5-30% missing: median + add binary indicator column
   - Numeric with >30% missing: consider dropping the feature
   - Categorical: mode, or "missing" as a new category if >5% missing

3. TRANSFORMATIONS:
   - Log transform: ONLY for right-skewed features (skewness > 1.0) with all positive values
   - Log1p transform: For right-skewed features that contain zeros
   - No transform needed for features going into tree-based models unless extremely skewed (>3.0)

4. CLASS IMBALANCE (if applicable):
   - Ratio 3:1 to 10:1: use scale_pos_weight in XGBoost/LightGBM, class_weight='balanced' in RF
   - Ratio >10:1: same as above + recommend appropriate metric (F1, PR-AUC over ROC-AUC)

5. METRIC SELECTION:
   - Balanced classification: accuracy or ROC-AUC
   - Imbalanced classification: F1-macro or PR-AUC
   - Regression: RMSE (if outliers matter) or MAE (if robust needed)

6. MODEL CONFIGURATION:
   - Small dataset (<1000 rows): fewer estimators (100), higher regularization
   - Medium dataset (1000-50000): default params work well
   - Large dataset (>50000): can use more estimators, lower learning rate

Return ONLY this JSON structure:
{{
  "reasoning": "2-3 sentences explaining your overall strategy",
  "target_info": {{"type": "classification/regression"}},
  "data_preprocessing": [
    {{"action": "handle_missing", "column": "col_name", "strategy": "median/mode/drop_feature/indicator", "reason": "brief why"}}
  ],
  "feature_engineering": [
    {{"action": "target_encode/onehot_encode/label_encode/frequency_encode/log_transform/log1p_transform/drop", "column": "col_name", "reason": "brief why"}}
  ],
  "encoding_rationale": "1-2 sentences explaining your encoding philosophy for this dataset",
  "baseline_models": ["lightgbm", "xgboost", "random_forest"],
  "model_params": {{
    "lightgbm": {{"n_estimators": 300, "learning_rate": 0.05, "extra_params": {{}}}},
    "xgboost": {{"n_estimators": 300, "learning_rate": 0.05, "extra_params": {{}}}},
    "random_forest": {{"n_estimators": 200, "extra_params": {{}}}}
  }},
  "evaluation": {{
    "metric": "the best metric for this problem",
    "reason": "why this metric"
  }},
  "expected_challenges": ["list potential issues"]
}}"""

    messages = [
        {"role": "system", "content": "You are an expert ML engineer. Return ONLY valid JSON. No markdown, no explanation outside the JSON."},
        {"role": "user", "content": prompt}
    ]

    response_text = _call_groq(messages, temperature=0.2, max_tokens=3000)
    plan = _parse_json_response(response_text)

    if plan:
        _plan_cache[profile_hash] = plan
        return plan

    # Fallback: rule-based plan if LLM fails
    print("⚠️ LLM failed, using rule-based fallback plan")
    return _fallback_plan(profile)


def chat_with_context(user_message, context):
    """
    Chat interface for asking questions about the model/results.
    Keeps context compact to stay within token limits.
    """
    # Compact the context to save tokens
    compact_context = json.dumps(context, indent=1, default=str)
    # Truncate if too long
    if len(compact_context) > 3000:
        compact_context = compact_context[:3000] + "..."

    messages = [
        {"role": "system", "content": "You are a helpful ML assistant. Answer concisely based on the provided context. If you don't know, say so."},
        {"role": "user", "content": f"Context:\n{compact_context}\n\nQuestion: {user_message}"}
    ]

    response = _call_groq(messages, temperature=0.4, max_tokens=1000)
    return response if response else "Sorry, I couldn't generate a response. Please try again."


def _compact_profile(profile):
    """
    Reduce profile size to fit within token limits.
    Keep only what the LLM needs to make decisions.
    """
    compact = {
        "rows": profile["basic_info"]["n_rows"],
        "target": profile["basic_info"]["target_column"],
        "target_type": profile["target_info"].get("type", "unknown"),
        "features": [],
    }

    # Class balance info
    if profile.get("class_balance"):
        compact["class_balance"] = profile["class_balance"]

    # Feature details (compact format)
    for f in profile["feature_details"]:
        feat = {"name": f["name"], "type": f["type"]}
        if f.get("null_pct", 0) > 0:
            feat["null_pct"] = f["null_pct"]
        if f["type"] == "numeric":
            if "skewness" in f:
                feat["skew"] = f["skewness"]
            if f.get("likely_categorical"):
                feat["likely_cat"] = True
            if f.get("has_negative"):
                feat["has_neg"] = True
        else:
            feat["cardinality"] = f.get("cardinality", f.get("unique", 0))
        compact["features"].append(feat)

    # Top correlations
    if profile.get("correlations", {}).get("high_pairs"):
        compact["high_correlations"] = profile["correlations"]["high_pairs"][:5]

    if profile.get("correlations", {}).get("potential_leakage"):
        compact["potential_leakage"] = profile["correlations"]["potential_leakage"]

    # Mutual info top 5
    if profile.get("correlations", {}).get("mutual_info_top10"):
        compact["top_predictive_features"] = profile["correlations"]["mutual_info_top10"][:5]

    compact["dataset_size"] = profile["recommendations_context"]["dataset_size"]

    return compact


def _fallback_plan(profile):
    """
    Rule-based fallback plan when LLM is unavailable.
    Better than nothing — applies sensible defaults.
    """
    target_type = profile["target_info"].get("type", "classification")
    is_imbalanced = profile.get("class_balance", {}).get("is_imbalanced", False)

    preprocessing = []
    feature_eng = []

    for f in profile["feature_details"]:
        # Handle missing
        if f.get("null_pct", 0) > 0:
            if f["type"] == "numeric":
                strategy = "median" if f["null_pct"] < 30 else "drop_feature"
            else:
                strategy = "mode"
            preprocessing.append({
                "action": "handle_missing",
                "column": f["name"],
                "strategy": strategy,
                "reason": f"Has {f['null_pct']}% missing values"
            })

        # Encoding decisions
        if f["type"] == "categorical":
            cardinality = f.get("cardinality", f.get("unique", 0))
            if cardinality <= 10:
                feature_eng.append({
                    "action": "onehot_encode",
                    "column": f["name"],
                    "reason": f"Low cardinality ({cardinality}), one-hot is safe"
                })
            elif cardinality <= 50:
                feature_eng.append({
                    "action": "target_encode",
                    "column": f["name"],
                    "reason": f"Medium cardinality ({cardinality}), target encoding preserves info"
                })
            else:
                feature_eng.append({
                    "action": "frequency_encode",
                    "column": f["name"],
                    "reason": f"High cardinality ({cardinality}), frequency encoding is safe"
                })

        # Log transform for skewed
        if f["type"] == "numeric" and abs(f.get("skewness", 0)) > 2.0:
            action = "log1p_transform" if f.get("has_negative") or f.get("zeros_pct", 0) > 0 else "log_transform"
            feature_eng.append({
                "action": action,
                "column": f["name"],
                "reason": f"High skewness ({f.get('skewness', 0)})"
            })

    # Metric selection
    if target_type == "classification":
        metric = "f1" if is_imbalanced else "roc_auc"
    else:
        metric = "rmse"

    return {
        "reasoning": "Fallback rule-based plan (LLM unavailable). Applied standard encoding rules and skewness corrections.",
        "target_info": {"type": target_type},
        "data_preprocessing": preprocessing,
        "feature_engineering": feature_eng,
        "encoding_rationale": "One-hot for low cardinality, target encoding for medium, frequency for high cardinality.",
        "baseline_models": ["lightgbm", "xgboost", "random_forest"],
        "model_params": {
            "lightgbm": {"n_estimators": 300, "learning_rate": 0.05, "extra_params": {}},
            "xgboost": {"n_estimators": 300, "learning_rate": 0.05, "extra_params": {}},
            "random_forest": {"n_estimators": 200, "extra_params": {}}
        },
        "evaluation": {"metric": metric, "reason": "Auto-selected based on problem type and class balance"},
        "expected_challenges": []
    }
