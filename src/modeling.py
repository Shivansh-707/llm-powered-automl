import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
import shap
from config import CV_FOLDS, RANDOM_STATE, SHAP_SAMPLE_SIZE, SHAP_TOP_FEATURES


class BaselineModeler:
    """
    Expert baseline model trainer with:
    - Class imbalance handling
    - Smart hyperparameters based on dataset size
    - SHAP-based feature importance
    - Stratified CV for classification
    """

    def __init__(self, problem_type='classification'):
        self.problem_type = problem_type
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = -np.inf
        self.shap_values = None
        self.shap_feature_names = None

    def get_models(self, n_rows=None, imbalance_ratio=1.0, model_params=None):
        """
        Initialize models with intelligent hyperparameters.
        Adjusts based on dataset size and class imbalance.
        """
        # Determine params based on dataset size
        if n_rows and n_rows < 1000:
            n_est_boost = 150
            n_est_rf = 100
            lr = 0.1
        elif n_rows and n_rows > 50000:
            n_est_boost = 500
            n_est_rf = 300
            lr = 0.03
        else:
            n_est_boost = 300
            n_est_rf = 200
            lr = 0.05

        # Override with LLM-suggested params if available
        if model_params:
            lgbm_params = model_params.get('lightgbm', {})
            xgb_params = model_params.get('xgboost', {})
            rf_params = model_params.get('random_forest', {})
            n_est_boost = lgbm_params.get('n_estimators', n_est_boost)
            lr = lgbm_params.get('learning_rate', lr)

        if self.problem_type == 'classification':
            # Handle class imbalance
            scale_weight = imbalance_ratio if imbalance_ratio > 2 else 1.0

            models = {
                'lightgbm': LGBMClassifier(
                    random_state=RANDOM_STATE,
                    verbose=-1,
                    n_estimators=n_est_boost,
                    learning_rate=lr,
                    scale_pos_weight=scale_weight,
                    num_leaves=31,
                    min_child_samples=max(20, int(n_rows * 0.01)) if n_rows else 20,
                ),
                'xgboost': XGBClassifier(
                    random_state=RANDOM_STATE,
                    eval_metric='logloss',
                    n_estimators=n_est_boost,
                    learning_rate=lr,
                    scale_pos_weight=scale_weight,
                    max_depth=6,
                    verbosity=0,
                ),
                'random_forest': RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_estimators=n_est_rf,
                    class_weight='balanced' if imbalance_ratio > 2 else None,
                    max_depth=None,
                    min_samples_leaf=max(1, int(n_rows * 0.005)) if n_rows else 1,
                )
            }
        else:
            models = {
                'lightgbm': LGBMRegressor(
                    random_state=RANDOM_STATE,
                    verbose=-1,
                    n_estimators=n_est_boost,
                    learning_rate=lr,
                    num_leaves=31,
                ),
                'xgboost': XGBRegressor(
                    random_state=RANDOM_STATE,
                    n_estimators=n_est_boost,
                    learning_rate=lr,
                    max_depth=6,
                    verbosity=0,
                ),
                'random_forest': RandomForestRegressor(
                    random_state=RANDOM_STATE,
                    n_estimators=n_est_rf,
                    max_depth=None,
                )
            }

        return models

    def get_metric(self, metric_name):
        """Map metric names to sklearn scoring strings."""
        metric_map = {
            'accuracy': 'accuracy',
            'roc_auc': 'roc_auc',
            'f1': 'f1_macro',
            'f1_macro': 'f1_macro',
            'f1_binary': 'f1',
            'pr_auc': 'average_precision',
            'rmse': 'neg_root_mean_squared_error',
            'mae': 'neg_mean_absolute_error',
            'r2': 'r2',
            'mse': 'neg_mean_squared_error',
        }
        return metric_map.get(metric_name, 'accuracy')

    def train_baseline(self, X, y, models_to_train=None, metric='accuracy',
                       imbalance_ratio=1.0, model_params=None):
        """
        Train baseline models with proper CV strategy.
        Uses stratified CV for classification, regular for regression.
        """
        if models_to_train is None:
            models_to_train = ['lightgbm', 'xgboost', 'random_forest']

        n_rows = len(X)
        available_models = self.get_models(
            n_rows=n_rows,
            imbalance_ratio=imbalance_ratio,
            model_params=model_params
        )
        scoring = self.get_metric(metric)

        # Choose CV strategy
        if self.problem_type == 'classification':
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        else:
            cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        for model_name in models_to_train:
            if model_name not in available_models:
                continue

            model = available_models[model_name]

            try:
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()

                # Fit on full data for feature importance / SHAP
                model.fit(X, y)

                self.models[model_name] = model
                self.results[model_name] = {
                    'cv_mean': float(abs(mean_score)),  # abs for neg metrics
                    'cv_std': float(std_score),
                    'cv_scores': [float(abs(s)) for s in cv_scores],
                    'metric': metric,
                    'raw_cv_mean': float(mean_score),  # Keep sign for comparison
                }

                # For neg metrics (rmse, mae), higher raw = better (less negative)
                if mean_score > self.best_score:
                    self.best_score = mean_score
                    self.best_model = model_name

                display_score = abs(mean_score)
                print(f"✓ {model_name}: {metric} = {display_score:.4f} (±{std_score:.4f})")

            except Exception as e:
                print(f"✗ {model_name} failed: {str(e)}")
                self.results[model_name] = {'error': str(e)}

        return self.results

    def compute_shap_importance(self, X, feature_names=None):
        """
        Compute SHAP values for the best model.
        This tells us which features ACTUALLY matter, not just split-based importance.
        """
        if self.best_model is None or self.best_model not in self.models:
            return None

        model = self.models[self.best_model]

        # Sample for speed on large datasets
        if len(X) > SHAP_SAMPLE_SIZE:
            if isinstance(X, pd.DataFrame):
                X_sample = X.sample(n=SHAP_SAMPLE_SIZE, random_state=RANDOM_STATE)
            else:
                idx = np.random.RandomState(RANDOM_STATE).choice(len(X), SHAP_SAMPLE_SIZE, replace=False)
                X_sample = X[idx]
        else:
            X_sample = X

        try:
            # Use TreeExplainer for tree-based models (fast)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Multi-class: list of arrays, one per class
                # Take mean absolute across all classes
                mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            elif shap_values.ndim == 3:
                # Shape: (n_samples, n_features, n_classes)
                mean_shap = np.abs(shap_values).mean(axis=(0, 2))
            else:
                # Binary or regression: single 2D array
                mean_shap = np.abs(shap_values).mean(axis=0)

            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(mean_shap))]

            # Sort by importance
            indices = np.argsort(mean_shap)[::-1][:SHAP_TOP_FEATURES]

            self.shap_values = shap_values
            self.shap_feature_names = feature_names

            return {
                'features': [feature_names[i] for i in indices],
                'shap_importance': [float(mean_shap[i]) for i in indices],
                'method': 'SHAP (TreeExplainer)',
                'model_used': self.best_model,
            }

        except Exception as e:
            print(f"⚠️ SHAP computation failed: {e}")
            # Fallback to built-in feature importance
            return self.get_feature_importance(feature_names)

    def get_feature_importance(self, feature_names, top_n=15):
        """Fallback: get split-based feature importance from best model."""
        if self.best_model is None or self.best_model not in self.models:
            return None

        model = self.models[self.best_model]

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]

            return {
                'features': [feature_names[i] for i in indices],
                'importances': [float(importances[i]) for i in indices],
                'method': 'split-based (built-in)',
                'model_used': self.best_model,
            }

        return None

    def get_summary(self):
        """Get comprehensive summary of all models trained."""
        return {
            'problem_type': self.problem_type,
            'models_trained': list(self.results.keys()),
            'best_model': self.best_model,
            'best_score': float(abs(self.best_score)) if self.best_score != -np.inf else 0,
            'results': self.results
        }
