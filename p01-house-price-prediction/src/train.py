import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from dotenv import load_dotenv

from src import preprocess, features

load_dotenv()
SEED        = 42
TARGET      = 'SalePrice'
MODEL_PATH  = os.getenv('MODEL_PATH', 'models/house_price_model_tuned.pkl')

os.makedirs('models', exist_ok=True)


def main():
    train_df, test_df = preprocess.load_data()

    # target distribution plot & log-transform
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(train_df[TARGET], bins=50, color='steelblue', edgecolor='white')
    axes[0].set_title('SalePrice — Raw')
    axes[1].hist(np.log1p(train_df[TARGET]), bins=50, color='darkorange', edgecolor='white')
    axes[1].set_title('SalePrice — Log Transformed')
    plt.tight_layout()
    plt.savefig('data/processed/target_distribution.png')
    plt.show()

    train_df[TARGET] = np.log1p(train_df[TARGET])

    train_df, test_df, _ = preprocess.drop_high_missing(train_df, test_df)

    train_df, test_df = features.engineer_features(train_df, test_df)

    X_train, X_val, y_train, y_val, num_cols, cat_cols = preprocess.split_features_target(train_df)

    preprocessor = preprocess.build_preprocessor(num_cols, cat_cols)

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

    # hyperparameter tuning
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

    # evaluate on validation set (log scale)
    y_pred = best_model.predict(X_val)
    rmse   = np.sqrt(mean_squared_error(y_val, y_pred))
    mae    = mean_absolute_error(y_val, y_pred)
    r2     = r2_score(y_val, y_pred)

    print(f"\n── Validation Metrics ──")
    print(f"RMSE : {rmse:.4f} (log scale)")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")

    # save model
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    # sample inference function
    def predict_price(input_dict: dict, model_path: str = MODEL_PATH) -> float:
        model     = joblib.load(model_path)
        input_df  = pd.DataFrame([input_dict])
        log_pred  = model.predict(input_df)[0]
        return round(np.expm1(log_pred), 2)

    sample = X_val.iloc[0].to_dict()
    print(f"\nSample Prediction : ${predict_price(sample):,.2f}")
    print(f"Actual Price      : ${np.expm1(y_val.iloc[0]):,.2f}")


if __name__ == '__main__':
    from sklearn.pipeline import Pipeline
    main()
