import imblearn
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

def random_forest():
    df = pd.read_csv("../data/processed/train.csv")
    x = df.drop("loan_status", axis=1)
    y = df["loan_status"]
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf_pipeline = imblearn.pipeline.make_pipeline(
        StandardScaler(),
        SMOTE(sampling_strategy=0.5, random_state=42),
        RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    )

    rf_param_grid = {
        'randomforestclassifier__n_estimators': [100, 200, 300],
        'randomforestclassifier__max_depth': [10, 20, 30, None],
        'randomforestclassifier__min_samples_split': [2, 5, 10],
        'randomforestclassifier__min_samples_leaf': [1, 2, 4],
        'randomforestclassifier__max_features': ['sqrt', 'log2', 0.5, 0.7]
    }

    rf_gs = GridSearchCV(
        rf_pipeline,
        param_grid=rf_param_grid,
        cv=skf,
        scoring='average_precision',
        n_jobs=-1,
        verbose=1
    )

    rf_gs.fit(x_train, y_train)
    print(f'Лучшие параметры: {rf_gs.best_params_}')
    print(f'Лучшее качество на кросс-валидации: {rf_gs.best_score_:.3f}')

    y_pred_proba = rf_gs.predict_proba(x_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    print(f"ROC-AUC: {roc_auc:.4f}")

    y_pred = rf_gs.predict(x_val)
    report = classification_report(y_val, y_pred)
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    random_forest()