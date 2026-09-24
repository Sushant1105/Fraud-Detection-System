"""
Data preprocessing and feature engineering module for Real-Time Fraud Detection System.
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd


class TransactionPreprocessor(BaseEstimator, TransformerMixin):
    """
    Production-grade preprocessor for financial fraud transaction features.
    
    Engineering & Transformation Logic:
    - V1 to V28: Preserved as-is (already centered, standardized PCA components).
    - Amount:
        * scaled_amount: Scaled using RobustScaler (subtracts median, divides by IQR).
        * log_amount: log1p(Amount) compressing long right-tail skewness.
    - Time:
        * scaled_time: Scaled using RobustScaler.
        * hour: Hour of day [0, 23].
        * hour_sin / hour_cos: Continuous cyclical trigonometric coordinates.
        
    Leakage Prevention:
    - All scaler parameters (median, IQR) are learned strictly on training data during fit().
    - Never fitted on test data or the full dataset.
    """
    def __init__(self):
        self.amount_scaler = RobustScaler()
        self.time_scaler = RobustScaler()
        self.is_fitted = False
        self.feature_names = None

    def fit(self, X, y=None):
        """Fit internal scalers strictly on training data."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected pandas DataFrame as input.")
            
        self.amount_scaler.fit(X[['Amount']])
        self.time_scaler.fit(X[['Time']])
        self.is_fitted = True
        return self

    def transform(self, X):
        """Transform raw input features into engineered, scaled feature DataFrame."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted. Call fit() before transform().")
            
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected pandas DataFrame as input.")
            
        X_df = X.copy()
        
        # 1. Temporal Feature Engineering
        hour = (X_df['Time'] // 3600) % 24
        hour_rad = 2.0 * np.pi * hour / 24.0
        
        # 2. Scaled & Derived Features
        scaled_amount = self.amount_scaler.transform(X_df[['Amount']]).flatten()
        scaled_time = self.time_scaler.transform(X_df[['Time']]).flatten()
        log_amount = np.log1p(np.maximum(X_df['Amount'], 0.0)).values
        hour_sin = np.sin(hour_rad).values
        hour_cos = np.cos(hour_rad).values
        
        # 3. Assemble Output DataFrame
        pca_cols = [f'V{i}' for i in range(1, 29)]
        transformed = pd.DataFrame(index=X_df.index)
        
        for col in pca_cols:
            transformed[col] = X_df[col].values
            
        transformed['scaled_amount'] = scaled_amount
        transformed['log_amount'] = log_amount
        transformed['scaled_time'] = scaled_time
        transformed['hour'] = hour.values
        transformed['hour_sin'] = hour_sin
        transformed['hour_cos'] = hour_cos
        
        if self.feature_names is None:
            self.feature_names = transformed.columns.tolist()
            
        return transformed
