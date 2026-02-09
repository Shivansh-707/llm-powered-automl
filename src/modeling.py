import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from config import CV_FOLDS, RANDOM_STATE

class BaselineModeler:
    """Train and evaluate baseline models."""
    
    def __init__(self, problem_type='classification'):
        self.problem_type = problem_type
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = -np.inf if problem_type == 'classification' else np.inf
    
    def get_models(self):
        """Initialize baseline models based on problem type."""
        if self.problem_type == 'classification':
            return {
                'lightgbm': LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
                'xgboost': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'),
                'random_forest': RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100)
            }
        else:
            return {
                'lightgbm': LGBMRegressor(random_state=RANDOM_STATE, verbose=-1),
                'xgboost': XGBRegressor(random_state=RANDOM_STATE),
                'random_forest': RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=100)
            }
    
    def get_metric(self, metric_name):
        """Get sklearn scoring metric."""
        metric_map = {
            'accuracy': 'accuracy',
            'roc_auc': 'roc_auc',
            'f1': 'f1',
            'rmse': 'neg_root_mean_squared_error',
            'mae': 'neg_mean_absolute_error',
            'r2': 'r2'
        }
        return metric_map.get(metric_name, 'accuracy')
    
    def train_baseline(self, X, y, models_to_train=None, metric='accuracy'):
        """Train baseline models with cross-validation."""
        if models_to_train is None:
            models_to_train = ['lightgbm', 'xgboost', 'random_forest']
        
        available_models = self.get_models()
        scoring = self.get_metric(metric)
        
        for model_name in models_to_train:
            if model_name not in available_models:
                continue
            
            model = available_models[model_name]
            
            try:
                cv_scores = cross_val_score(model, X, y, cv=CV_FOLDS, scoring=scoring)
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                
                model.fit(X, y)
                
                self.models[model_name] = model
                self.results[model_name] = {
                    'cv_mean': float(mean_score),
                    'cv_std': float(std_score),
                    'cv_scores': cv_scores.tolist(),
                    'metric': metric
                }
                
                is_better = (
                    (self.problem_type == 'classification' and mean_score > self.best_score) or
                    (self.problem_type == 'regression' and mean_score > self.best_score)
                )
                
                if is_better:
                    self.best_score = mean_score
                    self.best_model = model_name
                
                print(f"✓ {model_name}: {metric} = {mean_score:.4f} (+/- {std_score:.4f})")
                
            except Exception as e:
                print(f"✗ {model_name} failed: {str(e)}")
                self.results[model_name] = {'error': str(e)}
        
        return self.results
    
    def get_feature_importance(self, feature_names, top_n=10):
        """Get feature importance from best model."""
        if self.best_model is None or self.best_model not in self.models:
            return None
        
        model = self.models[self.best_model]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            return {
                'features': [feature_names[i] for i in indices],
                'importances': [float(importances[i]) for i in indices]
            }
        
        return None
    
    def get_summary(self):
        """Get summary of all models trained."""
        return {
            'problem_type': self.problem_type,
            'models_trained': list(self.results.keys()),
            'best_model': self.best_model,
            'best_score': float(self.best_score),
            'results': self.results
        }
