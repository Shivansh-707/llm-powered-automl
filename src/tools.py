import pandas as pd
import numpy as np

class AutoMLTools:
    """Tools for function calling / chat interface."""
    
    def __init__(self, modeler, feature_engineer, df, target_col):
        self.modeler = modeler
        self.feature_engineer = feature_engineer
        self.df = df
        self.target_col = target_col
    
    def get_feature_importance(self, top_n=10):
        """Get top N most important features."""
        if self.modeler.best_model is None:
            return "No model has been trained yet."
        
        feature_names = [col for col in self.df.columns if col != self.target_col]
        importance = self.modeler.get_feature_importance(feature_names, top_n)
        
        if importance is None:
            return "Feature importance not available for this model."
        
        result = f"Top {top_n} Important Features:\n"
        for feat, imp in zip(importance['features'], importance['importances']):
            result += f"- {feat}: {imp:.4f}\n"
        
        return result
    
    def get_model_scores(self):
        """Get scores for all trained models."""
        if not self.modeler.results:
            return "No models have been trained yet."
        
        result = "Model Performance:\n"
        for model_name, metrics in self.modeler.results.items():
            if 'error' in metrics:
                result += f"- {model_name}: Failed ({metrics['error']})\n"
            else:
                result += f"- {model_name}: {metrics['metric']} = {metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f})\n"
        
        result += f"\nBest Model: {self.modeler.best_model}"
        return result
    
    def get_dataset_info(self):
        """Get basic dataset information."""
        info = f"""Dataset Information:
- Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns
- Target: {self.target_col}
- Features: {', '.join([col for col in self.df.columns if col != self.target_col])}
- Missing Values: {self.df.isnull().sum().sum()} cells
- Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
"""
        return info
    
    def get_transformations_applied(self):
        """Get list of all feature engineering transformations."""
        summary = self.feature_engineer.get_summary()
        
        if summary['total_transformations'] == 0:
            return "No transformations have been applied yet."
        
        result = f"Total Transformations: {summary['total_transformations']}\n\n"
        for trans in summary['transformations']:
            result += f"- {trans}\n"
        
        return result
