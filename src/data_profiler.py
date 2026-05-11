import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def profile_dataset(df, target_col):
    """
    Expert-level dataset profiling.
    Produces a rich, compact profile that gives the LLM everything it needs
    to make intelligent decisions in a single prompt.
    """
    profile = {
        "basic_info": {
            "n_rows": len(df),
            "n_features": len(df.columns) - 1,
            "target_column": target_col,
        },
        "target_info": {},
        "feature_details": [],
        "correlations": {},
        "data_quality": {},
        "class_balance": {},
        "recommendations_context": {}
    }

    # --- Target Analysis ---
    if target_col in df.columns:
        target_series = df[target_col]
        target_unique = target_series.nunique()
        is_classification = target_unique <= 20

        profile["target_info"] = {
            "unique_values": int(target_unique),
            "type": "classification" if is_classification else "regression",
            "null_count": int(target_series.isnull().sum()),
        }

        if is_classification:
            value_counts = target_series.value_counts()
            profile["target_info"]["class_distribution"] = {
                str(k): int(v) for k, v in value_counts.items()
            }
            majority = value_counts.iloc[0]
            minority = value_counts.iloc[-1]
            imbalance_ratio = float(round(majority / minority, 2)) if minority > 0 else float('inf')
            profile["class_balance"] = {
                "majority_class_count": int(majority),
                "minority_class_count": int(minority),
                "imbalance_ratio": imbalance_ratio,
                "is_imbalanced": bool(imbalance_ratio > 3)
            }
        else:
            profile["target_info"]["stats"] = {
                "mean": round(float(target_series.mean()), 4),
                "std": round(float(target_series.std()), 4),
                "skewness": round(float(stats.skew(target_series.dropna())), 4),
            }

    # --- Feature-by-Feature Analysis ---
    numeric_cols = []
    categorical_cols = []

    for col in df.columns:
        if col == target_col:
            continue

        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "null_pct": round(df[col].isnull().sum() / len(df) * 100, 1),
            "unique": int(df[col].nunique()),
        }

        # Numeric features
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            col_info["type"] = "numeric"
            skewness = float(stats.skew(df[col].dropna()))
            col_info["skewness"] = round(skewness, 3)
            col_info["has_negative"] = bool(float(df[col].min()) < 0)
            col_info["zeros_pct"] = round(float((df[col] == 0).sum()) / len(df) * 100, 1)

            # Flag numeric columns that might actually be categorical
            if col_info["unique"] <= 10:
                col_info["likely_categorical"] = True

            numeric_cols.append(col)
        else:
            col_info["type"] = "categorical"
            col_info["cardinality"] = int(df[col].nunique())
            col_info["top_3"] = list(df[col].value_counts().head(3).index)

            # Detect potential date columns
            if _looks_like_date(df[col]):
                col_info["likely_date"] = True

            categorical_cols.append(col)

        profile["feature_details"].append(col_info)

    # --- Correlation Analysis (top pairs only, to save tokens) ---
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        # Get top 5 correlated pairs (excluding self-correlation)
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        top_corr_pairs = []
        for col in upper_tri.columns:
            for idx in upper_tri.index:
                val = upper_tri.loc[idx, col]
                if pd.notna(val) and val > 0.5:
                    top_corr_pairs.append({
                        "feature_1": idx,
                        "feature_2": col,
                        "correlation": round(val, 3)
                    })

        top_corr_pairs.sort(key=lambda x: x["correlation"], reverse=True)
        profile["correlations"]["high_pairs"] = top_corr_pairs[:8]

        # Check for potential leakage (features > 0.95 correlated with target)
        if target_col in df.columns and df[target_col].dtype in ['int64', 'float64']:
            target_corr = df[numeric_cols].corrwith(df[target_col]).abs()
            leakage_suspects = target_corr[target_corr > 0.95].index.tolist()
            if leakage_suspects:
                profile["correlations"]["potential_leakage"] = leakage_suspects

    # --- Mutual Information (feature relevance to target) ---
    if target_col in df.columns and len(numeric_cols) > 0:
        try:
            X_numeric = df[numeric_cols].fillna(0)
            y = df[target_col]
            if profile["target_info"]["type"] == "classification":
                mi_scores = mutual_info_classif(X_numeric, y, random_state=42)
            else:
                mi_scores = mutual_info_regression(X_numeric, y, random_state=42)

            mi_ranked = sorted(
                zip(numeric_cols, mi_scores),
                key=lambda x: x[1], reverse=True
            )
            profile["correlations"]["mutual_info_top10"] = [
                {"feature": name, "mi_score": round(float(score), 4)}
                for name, score in mi_ranked[:10]
            ]
        except Exception:
            pass  # MI can fail on edge cases, not critical

    # --- Data Quality Summary ---
    constant_cols = [col for col in df.columns if col != target_col and df[col].nunique() <= 1]
    profile["data_quality"] = {
        "total_missing_pct": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
        "features_with_missing": int((df.drop(columns=[target_col]).isnull().sum() > 0).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_features": constant_cols,
    }

    # --- Recommendations Context ---
    profile["recommendations_context"] = {
        "has_high_cardinality": bool(any(
            f.get("cardinality", 0) > 50 for f in profile["feature_details"]
        )),
        "has_missing_values": bool(profile["data_quality"]["total_missing_pct"] > 0),
        "has_skewed_features": bool(any(
            abs(f.get("skewness", 0)) > 1.0 for f in profile["feature_details"]
        )),
        "has_class_imbalance": bool(profile["class_balance"].get("is_imbalanced", False)),
        "n_numeric": len(numeric_cols),
        "n_categorical": len(categorical_cols),
        "dataset_size": "small" if len(df) < 1000 else "medium" if len(df) < 50000 else "large",
    }

    return profile


def _looks_like_date(series, sample_size=20):
    """Heuristic check if a string column might be a date."""
    import warnings
    sample = series.dropna().head(sample_size)
    if len(sample) == 0:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pd.to_datetime(sample, format='mixed')
        return True
    except (ValueError, TypeError):
        return False
