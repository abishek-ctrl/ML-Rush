import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from dotenv import load_dotenv

from src import preprocess, features

load_dotenv()
SEED        = 42
TARGET      = 'SalePrice'
MODEL_PATH  = os.getenv('MODEL_PATH', 'models/house_price_model_tuned.pkl')


def main(model_path: str = MODEL_PATH):
    os.makedirs('data/processed', exist_ok=True)
    train_df, _ = preprocess.load_data(TRAIN_PATH, TRAIN_PATH)

    # log-transform target as in training
    train_df[TARGET] = np.log1p(train_df[TARGET])

    train_df, _ = preprocess.drop_high_missing(train_df, train_df)
    train_df, _ = features.engineer_features(train_df, train_df)

    _, X_val, _, y_val, _, _ = preprocess.split_features_target(train_df)

    model = joblib.load(model_path)

    y_pred = model.predict(X_val)
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

    # Feature importance
    gb_model      = model.named_steps['model']
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
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


if __name__ == '__main__':
    main()
