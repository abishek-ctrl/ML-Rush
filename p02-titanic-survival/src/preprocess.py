import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

SEED = 42
TARGET = 'Survived'


def load_data(train_path: str = None, test_path: str = None):
    """Load train/test CSV. Paths can be provided or read from environment (.env)."""
    load_dotenv()

    if train_path is None:
        train_path = os.getenv('TRAIN_PATH', 'data/raw/train.csv')
    if test_path is None:
        test_path = os.getenv('TEST_PATH', 'data/raw/test.csv')

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Train: {train_df.shape} | Test: {test_df.shape}")
    print(f"\nClass balance:\n{train_df[TARGET].value_counts(normalize=True).round(3)}")
    os.makedirs('data/processed', exist_ok=True)
    return train_df, test_df


def split_features_target(train_df: pd.DataFrame):
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET]

    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"Numerical : {num_cols}")
    print(f"Categorical: {cat_cols}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
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
