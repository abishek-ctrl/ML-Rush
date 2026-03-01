import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve
import joblib
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline

from src import preprocess, features

load_dotenv()
SEED         = 42
TARGET       = 'Survived'
MODEL_PATH   = os.getenv('MODEL_PATH', 'models/titanic_model.pkl')
THRESH_PATH  = os.getenv('THRESHOLD_PATH', 'models/titanic_threshold.pkl')

os.makedirs('models', exist_ok=True)


def main():
    train_df, test_df = preprocess.load_data()

    train_df, test_df = features.engineer_features(train_df, test_df)

    X_train, X_val, y_train, y_val, num_cols, cat_cols = preprocess.split_features_target(train_df)

    preprocessor = preprocess.build_preprocessor(num_cols, cat_cols)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # hyperparameter tuning - RandomForest
    param_grid = {
        'model__n_estimators'     : [200, 300, 500],
        'model__max_depth'        : [4, 6, 8, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf' : [1, 2, 4],
        'model__max_features'     : ['sqrt', 'log2'],
        'model__class_weight'     : ['balanced', None]
    }

    rf_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(random_state=SEED))
    ])

    search = RandomizedSearchCV(
        rf_pipe,
        param_distributions=param_grid,
        n_iter=40,
        scoring='roc_auc',
        cv=skf,
        random_state=SEED,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)

    print(f"\nBest Params  : {search.best_params_}")
    print(f"Best ROC-AUC : {search.best_score_:.4f}")

    # Voting Ensemble
    voting_clf = VotingClassifier(
        estimators=[
            ('lr',  LogisticRegression(max_iter=1000, random_state=SEED)),
            ('svm', SVC(probability=True, random_state=SEED)),
            ('rf',  search.best_estimator_.named_steps['model'])
        ],
        voting='soft'
    )

    voting_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', voting_clf)
    ])
    voting_pipe.fit(X_train, y_train)

    y_pred_prob_v = voting_pipe.predict_proba(X_val)[:, 1]

    # Threshold Tuning
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_prob_v)
    f1_scores   = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_thresh = thresholds[np.argmax(f1_scores)]

    print(f"\nDefault threshold (0.50) F1 : {f1_score(y_val, y_pred_prob_v >= 0.50):.4f}")
    print(f"Optimal threshold ({best_thresh:.2f}) F1 : {f1_score(y_val, y_pred_prob_v >= best_thresh):.4f}")

    joblib.dump(voting_pipe, MODEL_PATH)
    joblib.dump(best_thresh, THRESH_PATH)
    print(f"\nModel saved   → {MODEL_PATH}")
    print(f"Threshold saved → {THRESH_PATH}")


if __name__ == '__main__':
    main()
