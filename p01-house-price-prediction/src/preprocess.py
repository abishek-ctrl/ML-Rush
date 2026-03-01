import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

SEED = 42
TARGET = 'SalePrice'


def load_data(train_path: str = None, test_path: str = None):
    """Load train/test CSV. Paths can be provided or read from environment (.env).

    If no paths are provided, this will attempt to load `.env` and read
    `TRAIN_PATH` and `TEST_PATH` variables. Defaults fallback to `data/train.csv`
    and `data/test.csv`.
    """
    load_dotenv()

    if train_path is None:
        train_path = os.getenv('TRAIN_PATH', 'data/train.csv')
    if test_path is None:
        test_path = os.getenv('TEST_PATH', 'data/test.csv')

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Train: {train_df.shape} | Test: {test_df.shape}")
    os.makedirs('data/processed', exist_ok=True)
    return train_df, test_df


def drop_high_missing(train_df: pd.DataFrame, test_df: pd.DataFrame, threshold: float = 40.0):
    missing = train_df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_pct = (missing / len(train_df) * 100).round(2)
    missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
    print("\nTop 20 Missing Columns:")
    print(missing_df.head(20))

    high_missing = missing_pct[missing_pct > threshold].index.tolist()
    print(f"\nDropping high-missing columns: {high_missing}")
    train_df = train_df.drop(columns=high_missing)
    test_df = test_df.drop(columns=high_missing)
    return train_df, test_df, high_missing


def split_features_target(train_df: pd.DataFrame):
    X = train_df.drop(columns=[TARGET, 'Id'])
    y = train_df[TARGET]

    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"Numerical: {len(num_cols)} | Categorical: {len(cat_cols)}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
    return X_train, X_val, y_train, y_val, num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    return preprocessor
