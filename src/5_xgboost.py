import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import pandas as pd

from src.plotting import plot_confusion_matrix


def xgboost():
    df = pd.read_csv("../data/processed/train.csv")
    x = df.drop("loan_status", axis=1)
    y = df["loan_status"]
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    xgb_pipeline = imblearn.pipeline.make_pipeline(
        StandardScaler(),
        SMOTE(sampling_strategy=0.5, random_state=42),
        xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            tree_method='hist'
        )
    )

    xgb_param_grid = {
        'xgbclassifier__n_estimators': [100, 200, 300],
        'xgbclassifier__max_depth': [3, 5, 7, 10],
        'xgbclassifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'xgbclassifier__subsample': [0.7, 0.8, 0.9, 1.0],
        'xgbclassifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'xgbclassifier__gamma': [0, 0.1, 0.2, 0.5],
        'xgbclassifier__reg_alpha': [0, 0.01, 0.1, 1],
        'xgbclassifier__reg_lambda': [1, 1.5, 2, 3]
    }

    xgb_rsearch = RandomizedSearchCV(
        xgb_pipeline,
        param_distributions=xgb_param_grid,
        n_iter=15,
        cv=skf,
        scoring='average_precision',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    xgb_rsearch.fit(x_train, y_train)
    print(f'Лучшие параметры: {xgb_rsearch.best_params_}')
    print(f'Лучшее качество на кросс-валидации: {xgb_rsearch.best_score_:.3f}')

    y_pred_proba = xgb_rsearch.predict_proba(x_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    print(f"ROC-AUC: {roc_auc:.4f}")

    # y_pred = lgb_rsearch.predict(x_val)
    y_pred_thresh = (y_pred_proba >= 0.7).astype(int)
    report = classification_report(y_val, y_pred_thresh)
    print("\nClassification Report:")
    print(report)

    plot_confusion_matrix(y_val, y_pred_thresh, save_path="../images/xgboost/confusion_matrix.png")



if __name__ == "__main__":
    xgboost()