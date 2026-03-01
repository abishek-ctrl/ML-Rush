import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
SEED        = 42
TARGET      = 'SalePrice'
TRAIN_PATH  = 'data/raw/train.csv'
TEST_PATH   = 'data/raw/test.csv'
MODEL_PATH  = 'models/house_price_model_tuned.pkl'

os.makedirs('models', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
print(f"Train: {train_df.shape} | Test: {test_df.shape}")

# ── EDA: TARGET DISTRIBUTION ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(train_df[TARGET], bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('SalePrice — Raw')
axes[1].hist(np.log1p(train_df[TARGET]), bins=50, color='darkorange', edgecolor='white')
axes[1].set_title('SalePrice — Log Transformed')
plt.tight_layout()
plt.savefig('data/processed/target_distribution.png')
plt.show()

# log-transform target (fixes right skew)
train_df[TARGET] = np.log1p(train_df[TARGET])

# ── MISSING VALUE ANALYSIS ────────────────────────────────────────────────────
missing     = train_df.isnull().sum()
missing     = missing[missing > 0].sort_values(ascending=False)
missing_pct = (missing / len(train_df) * 100).round(2)
missing_df  = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print("\nTop 20 Missing Columns:")
print(missing_df.head(20))

# Drop columns with >40% missing
high_missing = missing_pct[missing_pct > 40].index.tolist()
print(f"\nDropping high-missing columns: {high_missing}")
train_df.drop(columns=high_missing, inplace=True)
test_df.drop(columns=high_missing, inplace=True)

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
for df in [train_df, test_df]:
    df['HouseAge']     = df['YrSold'] - df['YearBuilt']
    df['RemodAge']     = df['YrSold'] - df['YearRemodAdd']
    df['TotalSF']      = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBaths']   = (df['FullBath'] + df['BsmtFullBath'] +
                          0.5 * df['HalfBath'] + 0.5 * df['BsmtHalfBath'])
    df['HasGarage']    = (df['GarageArea'] > 0).astype(int)
    df['HasPool']      = (df['PoolArea'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)

print(f"\nAfter feature engineering — Train: {train_df.shape}")

# ── SPLIT FEATURES & TARGET ───────────────────────────────────────────────────
X = train_df.drop(columns=[TARGET, 'Id'])
y = train_df[TARGET]

cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"Numerical: {len(num_cols)} | Categorical: {len(cat_cols)}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

# ── PREPROCESSING PIPELINE ────────────────────────────────────────────────────
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# ── BASELINE MODEL COMPARISON ─────────────────────────────────────────────────
models = {
    'LinearRegression' : LinearRegression(),
    'Ridge'            : Ridge(alpha=10),
    'Lasso'            : Lasso(alpha=0.001),
    'RandomForest'     : RandomForestRegressor(n_estimators=200, random_state=SEED),
    'GradientBoosting' : GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=SEED)
}

kf      = KFold(n_splits=5, shuffle=True, random_state=SEED)
results = {}

print("\n── Baseline CV Results ──")
for name, model in models.items():
    pipe    = Pipeline([('preprocessor', preprocessor), ('model', model)])
    cv_rmse = np.sqrt(-cross_val_score(pipe, X_train, y_train,
                                       scoring='neg_mean_squared_error', cv=kf))
    results[name] = {'CV RMSE Mean': cv_rmse.mean().round(4),
                     'CV RMSE Std' : cv_rmse.std().round(4)}
    print(f"{name:25s} → RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")

# ── HYPERPARAMETER TUNING ─────────────────────────────────────────────────────
param_grid = {
    'model__n_estimators'    : [300, 500, 700],
    'model__learning_rate'   : [0.01, 0.05, 0.1],
    'model__max_depth'       : [3, 4, 5],
    'model__min_samples_leaf': [3, 5, 10],
    'model__subsample'       : [0.7, 0.8, 1.0]
}

tuned_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(random_state=SEED))
])

search = RandomizedSearchCV(
    tuned_pipe,
    param_distributions=param_grid,
    n_iter=30,
    scoring='neg_mean_squared_error',
    cv=kf,
    random_state=SEED,
    n_jobs=-1,
    verbose=1
)
search.fit(X_train, y_train)

print(f"\nBest Params : {search.best_params_}")
print(f"Best CV RMSE: {np.sqrt(-search.best_score_):.4f}")

best_model = search.best_estimator_

# ── EVALUATE ON VALIDATION SET ────────────────────────────────────────────────
y_pred = best_model.predict(X_val)
rmse   = np.sqrt(mean_squared_error(y_val, y_pred))
mae    = mean_absolute_error(y_val, y_pred)
r2     = r2_score(y_val, y_pred)

print(f"\n── Validation Metrics ──")
print(f"RMSE : {rmse:.4f} (log scale)")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

# Actual vs Predicted plot
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred, alpha=0.4, color='steelblue')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel('Actual log(SalePrice)')
plt.ylabel('Predicted log(SalePrice)')
plt.title('Actual vs Predicted — GradientBoosting Tuned')
plt.tight_layout()
plt.savefig('data/processed/actual_vs_predicted.png')
plt.show()

# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────
gb_model      = best_model.named_steps['model']
feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
importances   = pd.Series(gb_model.feature_importances_, index=feature_names)
top20         = importances.sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 7))
top20.plot(kind='barh', color='steelblue')
plt.gca().invert_yaxis()
plt.title('Top 20 Feature Importances')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('data/processed/feature_importance.png')
plt.show()

# ── SAVE MODEL ────────────────────────────────────────────────────────────────
joblib.dump(best_model, MODEL_PATH)
print(f"\nModel saved → {MODEL_PATH}")

# ── INFERENCE FUNCTION ────────────────────────────────────────────────────────
def predict_price(input_dict: dict, model_path: str = MODEL_PATH) -> float:
    model     = joblib.load(model_path)
    input_df  = pd.DataFrame([input_dict])
    log_pred  = model.predict(input_df)[0]
    return round(np.expm1(log_pred), 2)

sample = X_val.iloc[0].to_dict()
print(f"\nSample Prediction : ${predict_price(sample):,.2f}")
print(f"Actual Price      : ${np.expm1(y_val.iloc[0]):,.2f}")
