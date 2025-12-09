import imblearn
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

from src.plotting import plot_confusion_matrix, plot_all_metrics_bars


def lightgbm():
    df = pd.read_csv("../data/processed/train.csv")
    x = df.drop("loan_status", axis=1)
    y = df["loan_status"]
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    lgb_pipeline = imblearn.pipeline.make_pipeline(
        StandardScaler(),
        SMOTE(sampling_strategy=0.5, random_state=42),
        lgb.LGBMClassifier(
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            verbose=-1,
            metric='binary_logloss'
        )
    )

    lgb_param_grid = {
        'lgbmclassifier__n_estimators': [200, 300, 400],
        'lgbmclassifier__max_depth': [3, 5, 7],
        'lgbmclassifier__learning_rate': [0.01, 0.05, 0.1],
        'lgbmclassifier__num_leaves': [15, 31, 50],
        'lgbmclassifier__subsample': [0.7, 0.8, 0.9],
        'lgbmclassifier__colsample_bytree': [0.7, 0.8, 0.9],
        'lgbmclassifier__reg_alpha': [0.1, 0.5, 1],
        'lgbmclassifier__reg_lambda': [0.1, 0.5, 1],
        'lgbmclassifier__min_child_samples': [20, 50, 100],
        'smote__sampling_strategy': [0.3, 0.5, 0.7]
    }

    lgb_rsearch = RandomizedSearchCV(
        lgb_pipeline,
        param_distributions=lgb_param_grid,
        n_iter=50,
        cv=skf,
        scoring='average_precision',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    lgb_rsearch.fit(x_train, y_train)
    print(f'Лучшие параметры: {lgb_rsearch.best_params_}')
    print(f'Лучшее качество на кросс-валидации: {lgb_rsearch.best_score_:.3f}')

    y_pred_proba = lgb_rsearch.predict_proba(x_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    print(f"ROC-AUC: {roc_auc:.4f}")

    #y_pred = lgb_rsearch.predict(x_val)
    y_pred_thresh = (y_pred_proba >= 0.7).astype(int)
    report = classification_report(y_val, y_pred_thresh)
    print("\nClassification Report:")
    print(report)

    plot_confusion_matrix(y_val, y_pred_thresh, save_path="../images/lgbm/confusion_matrix.png")

    accuracy = accuracy_score(y_val, y_pred_thresh)
    precision = precision_score(y_val, y_pred_thresh, pos_label=1)
    recall = recall_score(y_val, y_pred_thresh, pos_label=1)
    f1 = f1_score(y_val, y_pred_thresh, pos_label=1)

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred_thresh).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics = {
        'ROC-AUC': roc_auc,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Specificity': specificity
    }
    plot_all_metrics_bars(metrics, model_name="LightGBM", save_path="../images/lgbm/metric_bars.png")



if __name__ == "__main__":
    lightgbm()
