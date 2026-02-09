import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from category_encoders import TargetEncoder
from config import MAX_CARDINALITY_ONEHOT, RANDOM_STATE

class FeatureEngineer:
    """Handles all feature engineering operations."""
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.transformations_applied = []
    
    def handle_missing(self, df, column, strategy='mean'):
        """Handle missing values in a column."""
        df = df.copy()
        
        if strategy == 'mean':
            df[column].fillna(df[column].mean(), inplace=True)
        elif strategy == 'median':
            df[column].fillna(df[column].median(), inplace=True)
        elif strategy == 'mode':
            df[column].fillna(df[column].mode()[0], inplace=True)
        elif strategy == 'drop':
            df.dropna(subset=[column], inplace=True)
        elif strategy == 'zero':
            df[column].fillna(0, inplace=True)
        
        self.transformations_applied.append(f"handle_missing:{column}:{strategy}")
        return df
    
    def target_encode(self, df, column, target_col):
        """Apply target encoding for high cardinality categorical features."""
        df = df.copy()
        
        if column not in self.encoders:
            encoder = TargetEncoder(cols=[column])
            df[column] = encoder.fit_transform(df[column], df[target_col])
            self.encoders[column] = encoder
        else:
            df[column] = self.encoders[column].transform(df[column])
        
        self.transformations_applied.append(f"target_encode:{column}")
        return df
    
    def onehot_encode(self, df, column):
        """Apply one-hot encoding for low cardinality categorical features."""
        df = df.copy()
        
        if df[column].nunique() <= MAX_CARDINALITY_ONEHOT:
            dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
            df = pd.concat([df.drop(column, axis=1), dummies], axis=1)
            self.transformations_applied.append(f"onehot_encode:{column}")
        else:
            print(f"Warning: {column} has too many unique values for one-hot encoding")
        
        return df
    
    def label_encode(self, df, column):
        """Apply label encoding for ordinal categorical features."""
        df = df.copy()
        
        if column not in self.encoders:
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column].astype(str))
            self.encoders[column] = encoder
        else:
            df[column] = self.encoders[column].transform(df[column].astype(str))
        
        self.transformations_applied.append(f"label_encode:{column}")
        return df
    
    def log_transform(self, df, column):
        """Apply log transformation for skewed numeric features."""
        df = df.copy()
        
        min_val = df[column].min()
        if min_val <= 0:
            df[column] = np.log1p(df[column] - min_val + 1)
        else:
            df[column] = np.log(df[column])
        
        self.transformations_applied.append(f"log_transform:{column}")
        return df
    
    def polynomial_features(self, df, columns, degree=2):
        """Create polynomial features from specified columns."""
        df = df.copy()
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(df[columns])
        
        feature_names = poly.get_feature_names_out(columns)
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
        
        new_features = [col for col in poly_df.columns if col not in columns]
        df = pd.concat([df, poly_df[new_features]], axis=1)
        
        self.transformations_applied.append(f"polynomial:{','.join(columns)}:degree_{degree}")
        return df
    
    def apply_plan(self, df, target_col, plan):
        """Apply full feature engineering plan from LLM."""
        df = df.copy()
        
        # Handle missing values first
        for step in plan.get('data_preprocessing', []):
            if step['action'] == 'handle_missing':
                df = self.handle_missing(df, step['column'], step['strategy'])
        
        # Apply feature engineering
        for step in plan.get('feature_engineering', []):
            action = step['action']
            column = step['column']
            
            if action == 'target_encode':
                df = self.target_encode(df, column, target_col)
            elif action == 'onehot_encode':
                df = self.onehot_encode(df, column)
            elif action == 'label_encode':
                df = self.label_encode(df, column)
            elif action == 'log_transform':
                df = self.log_transform(df, column)
            elif action == 'polynomial':
                degree = step.get('params', {}).get('degree', 2)
                columns = step.get('params', {}).get('columns', [column])
                df = self.polynomial_features(df, columns, degree)
        
        return df
    
    def get_summary(self):
        """Return summary of all transformations applied."""
        return {
            "total_transformations": len(self.transformations_applied),
            "transformations": self.transformations_applied
        }
