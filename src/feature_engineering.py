import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from category_encoders import TargetEncoder
from config import MAX_CARDINALITY_ONEHOT, RANDOM_STATE


class FeatureEngineer:
    """
    Expert feature engineering engine.
    Handles all encoding strategies with proper reasoning.
    """

    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.frequency_maps = {}
        self.transformations_applied = []
        self.encoding_decisions = []  # Track WHY each encoding was chosen

    def handle_missing(self, df, column, strategy='median'):
        """Handle missing values with multiple strategies."""
        df = df.copy()

        if column not in df.columns:
            return df

        # Normalize strategy string (LLM might say "median + indicator")
        strategy_lower = strategy.lower().strip()

        if 'indicator' in strategy_lower:
            # Fill with median AND add a binary indicator
            indicator_col = f"{column}_was_missing"
            df[indicator_col] = df[column].isnull().astype(int)
            df[column] = df[column].fillna(df[column].median() if df[column].dtype in ['int64', 'float64', 'int32', 'float32'] else df[column].mode().iloc[0])
            self.transformations_applied.append(f"handle_missing:{column}:median + indicator")
            return df

        if strategy_lower in ('mean',):
            df[column] = df[column].fillna(df[column].mean())
        elif strategy_lower in ('median',):
            df[column] = df[column].fillna(df[column].median())
        elif strategy_lower in ('mode',):
            df[column] = df[column].fillna(df[column].mode().iloc[0] if not df[column].mode().empty else 'missing')
        elif strategy_lower in ('drop',):
            df = df.dropna(subset=[column])
        elif strategy_lower in ('drop_feature',):
            df = df.drop(columns=[column])
        elif strategy_lower in ('zero',):
            df[column] = df[column].fillna(0)
        elif strategy_lower in ('missing_category', 'missing'):
            df[column] = df[column].fillna('MISSING')
        else:
            # Default to median for numeric, mode for categorical
            if df[column].dtype in ['int64', 'float64', 'int32', 'float32']:
                df[column] = df[column].fillna(df[column].median())
            else:
                df[column] = df[column].fillna(df[column].mode().iloc[0] if not df[column].mode().empty else 'missing')

        self.transformations_applied.append(f"handle_missing:{column}:{strategy}")
        return df

    def target_encode(self, df, column, target_col):
        """
        Target encoding for medium-to-high cardinality categoricals.
        Best when: cardinality > 10, enough rows to avoid overfitting.
        """
        df = df.copy()

        if column not in df.columns:
            return df

        if column not in self.encoders:
            encoder = TargetEncoder(cols=[column], smoothing=0.3)
            df[column] = encoder.fit_transform(df[[column]], df[target_col])
            self.encoders[column] = encoder
        else:
            df[column] = self.encoders[column].transform(df[[column]])

        self.transformations_applied.append(f"target_encode:{column}")
        self.encoding_decisions.append({
            "column": column,
            "encoding": "target",
            "reason": "Medium/high cardinality - target encoding captures relationship with target"
        })
        return df

    def onehot_encode(self, df, column):
        """
        One-hot encoding for low cardinality categoricals.
        Best when: ≤10 unique values, no ordinal relationship.
        """
        df = df.copy()

        if column not in df.columns:
            return df

        cardinality = df[column].nunique()
        if cardinality <= MAX_CARDINALITY_ONEHOT:
            dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
            # Ensure boolean columns become int
            dummies = dummies.astype(int)
            df = pd.concat([df.drop(column, axis=1), dummies], axis=1)
            self.transformations_applied.append(f"onehot_encode:{column} (created {cardinality-1} features)")
            self.encoding_decisions.append({
                "column": column,
                "encoding": "one-hot",
                "reason": f"Low cardinality ({cardinality}) - one-hot preserves all information without assumptions"
            })
        else:
            # Fallback to frequency encoding if cardinality too high
            print(f"⚠️ {column} has {cardinality} unique values, using frequency encoding instead")
            df = self.frequency_encode(df, column)

        return df

    def label_encode(self, df, column):
        """
        Label encoding for ordinal categoricals.
        Best when: values have a natural order (low/medium/high, small/large).
        """
        df = df.copy()

        if column not in df.columns:
            return df

        if column not in self.encoders:
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column].astype(str))
            self.encoders[column] = encoder
        else:
            # Handle unseen labels gracefully
            known_classes = set(self.encoders[column].classes_)
            df[column] = df[column].astype(str).apply(
                lambda x: self.encoders[column].transform([x])[0] if x in known_classes else -1
            )

        self.transformations_applied.append(f"label_encode:{column}")
        self.encoding_decisions.append({
            "column": column,
            "encoding": "label",
            "reason": "Ordinal feature - label encoding preserves natural order"
        })
        return df

    def frequency_encode(self, df, column):
        """
        Frequency encoding for high cardinality categoricals.
        Best when: very high cardinality (>50), small dataset where target encoding might overfit.
        Maps each category to its frequency in the training data.
        """
        df = df.copy()

        if column not in df.columns:
            return df

        if column not in self.frequency_maps:
            freq_map = df[column].value_counts(normalize=True).to_dict()
            self.frequency_maps[column] = freq_map
        else:
            freq_map = self.frequency_maps[column]

        df[column] = df[column].map(freq_map).fillna(0)

        self.transformations_applied.append(f"frequency_encode:{column}")
        self.encoding_decisions.append({
            "column": column,
            "encoding": "frequency",
            "reason": "High cardinality - frequency encoding is simple and doesn't overfit"
        })
        return df

    def log_transform(self, df, column):
        """Log transform for right-skewed features with all positive values."""
        df = df.copy()

        if column not in df.columns:
            return df

        min_val = df[column].min()
        if min_val <= 0:
            # Can't log negative/zero values, use log1p with shift
            df[column] = np.log1p(df[column] - min_val)
        else:
            df[column] = np.log(df[column])

        self.transformations_applied.append(f"log_transform:{column}")
        return df

    def log1p_transform(self, df, column):
        """Log1p transform for skewed features that may contain zeros."""
        df = df.copy()

        if column not in df.columns:
            return df

        min_val = df[column].min()
        if min_val < 0:
            df[column] = np.log1p(df[column] - min_val)
        else:
            df[column] = np.log1p(df[column])

        self.transformations_applied.append(f"log1p_transform:{column}")
        return df

    def polynomial_features(self, df, columns, degree=2):
        """Create polynomial/interaction features from specified columns."""
        df = df.copy()

        valid_cols = [c for c in columns if c in df.columns]
        if len(valid_cols) < 2:
            return df

        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
        poly_features = poly.fit_transform(df[valid_cols])

        feature_names = poly.get_feature_names_out(valid_cols)
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)

        new_features = [col for col in poly_df.columns if col not in valid_cols]
        df = pd.concat([df, poly_df[new_features]], axis=1)

        self.transformations_applied.append(f"interactions:{','.join(valid_cols)}")
        return df

    def drop_feature(self, df, column):
        """Drop a feature (constant, leaky, or too many missing values)."""
        df = df.copy()
        if column in df.columns:
            df = df.drop(columns=[column])
            self.transformations_applied.append(f"dropped:{column}")
        return df

    def apply_plan(self, df, target_col, plan):
        """
        Apply the full feature engineering plan from the LLM.
        Handles all action types with graceful error handling.
        """
        df = df.copy()

        # Step 1: Handle missing values
        for step in plan.get('data_preprocessing', []):
            action = step.get('action', '')
            column = step.get('column', '')

            if action == 'handle_missing' and column in df.columns:
                strategy = step.get('strategy', 'median')
                df = self.handle_missing(df, column, strategy)

        # Step 2: Apply feature engineering
        for step in plan.get('feature_engineering', []):
            action = step.get('action', '')
            column = step.get('column', '')

            if column and column not in df.columns:
                continue  # Skip if column was dropped or doesn't exist

            try:
                if action == 'target_encode':
                    df = self.target_encode(df, column, target_col)
                elif action == 'onehot_encode':
                    df = self.onehot_encode(df, column)
                elif action == 'label_encode':
                    df = self.label_encode(df, column)
                elif action == 'frequency_encode':
                    df = self.frequency_encode(df, column)
                elif action == 'log_transform':
                    df = self.log_transform(df, column)
                elif action == 'log1p_transform':
                    df = self.log1p_transform(df, column)
                elif action == 'drop':
                    df = self.drop_feature(df, column)
                elif action == 'polynomial':
                    degree = step.get('params', {}).get('degree', 2)
                    columns = step.get('params', {}).get('columns', [column])
                    df = self.polynomial_features(df, columns, degree)
            except Exception as e:
                print(f"⚠️ Failed to apply {action} on {column}: {e}")
                continue

        return df

    def get_summary(self):
        """Return detailed summary of all transformations."""
        return {
            "total_transformations": len(self.transformations_applied),
            "transformations": self.transformations_applied,
            "encoding_decisions": self.encoding_decisions
        }
