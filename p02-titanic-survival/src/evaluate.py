import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, RocCurveDisplay
import joblib
from dotenv import load_dotenv

from src import preprocess, features

load_dotenv()
SEED         = 42
TARGET       = 'Survived'
MODEL_PATH   = os.getenv('MODEL_PATH', 'models/titanic_model.pkl')
THRESH_PATH  = os.getenv('THRESHOLD_PATH', 'models/titanic_threshold.pkl')

def main(model_path: str = MODEL_PATH, thresh_path: str = THRESH_PATH):
    os.makedirs('data/processed', exist_ok=True)
    train_df, test_df = preprocess.load_data()
    train_df, test_df = features.engineer_features(train_df, test_df)

    _, X_val, _, y_val, _, _ = preprocess.split_features_target(train_df)

    model     = joblib.load(model_path)
    threshold = joblib.load(thresh_path)

    y_pred_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_prob >= threshold).astype(int)

    print("\n── Voting Ensemble Validation ──")
    print(classification_report(y_val, y_pred, target_names=['Died', 'Survived']))
    print(f"ROC-AUC: {roc_auc_score(y_val, y_pred_prob):.4f}")

    # Confusion Matrix + ROC Curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Died', 'Survived'], yticklabels=['Died', 'Survived'])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    RocCurveDisplay.from_predictions(y_val, y_pred_prob, ax=axes[1], color='steelblue')
    axes[1].set_title('ROC Curve')
    axes[1].plot([0, 1], [0, 1], 'r--')

    plt.tight_layout()
    plt.savefig('data/processed/eval_metrics.png')
    plt.show()

if __name__ == '__main__':
    main()
