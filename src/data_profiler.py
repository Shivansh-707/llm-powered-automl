import pandas as pd
import numpy as np
from scipy import stats

def profile_dataset(df, target_col):
    """
    Comprehensive dataset profiling for LLM analysis.
    Returns detailed statistics about the dataset.
    """
    profile = {
        "basic_info": {
            "n_rows": len(df),
            "n_features": len(df.columns),
            "target_column": target_col,
            "feature_names": list(df.columns)
        },
        "target_info": {},
        "feature_details": [],
        "data_quality": {},
        "recommendations_context": {}
    }
    
    # Target analysis
    if target_col in df.columns:
        target_unique = df[target_col].nunique()
        profile["target_info"] = {
            "unique_values": int(target_unique),
            "type": "classification" if target_unique < 20 else "regression",
            "null_count": int(df[target_col].isnull().sum()),
            "sample_values": df[target_col].value_counts().head(5).to_dict()
        }
    
    # Feature-by-feature analysis
    for col in df.columns:
        if col == target_col:
            continue
            
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "null_percentage": round(df[col].isnull().sum() / len(df) * 100, 2),
            "unique_values": int(df[col].nunique())
        }
        
        # Numeric features
        if df[col].dtype in ['int64', 'float64']:
            col_info["feature_type"] = "numeric"
            col_info["stats"] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "skewness": float(stats.skew(df[col].dropna()))
            }
            
            if col_info["unique_values"] < 10:
                col_info["might_be_categorical"] = True
        
        # Categorical features
        else:
            col_info["feature_type"] = "categorical"
            col_info["cardinality"] = int(df[col].nunique())
            col_info["top_values"] = df[col].value_counts().head(3).to_dict()
            
            if col_info["cardinality"] > 50:
                col_info["high_cardinality"] = True
        
        profile["feature_details"].append(col_info)
    
    # Data quality summary
    profile["data_quality"] = {
        "total_missing_cells": int(df.isnull().sum().sum()),
        "features_with_missing": int((df.isnull().sum() > 0).sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }
    
    # Context for recommendations
    profile["recommendations_context"] = {
        "has_high_cardinality_features": any(
            f.get("high_cardinality", False) for f in profile["feature_details"]
        ),
        "has_missing_values": profile["data_quality"]["total_missing_cells"] > 0,
        "n_numeric_features": len([f for f in profile["feature_details"] if f["feature_type"] == "numeric"]),
        "n_categorical_features": len([f for f in profile["feature_details"] if f["feature_type"] == "categorical"])
    }
    
    return profile
