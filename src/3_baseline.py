import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import (train_test_split, StratifiedKFold, GridSearchCV)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    classification_report
)
import imblearn

def baseline():
    df = pd.read_csv("../data/processed/train.csv")
    x = df.drop("loan_status", axis=1)
    y = df["loan_status"]
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        'logisticregression__l1_ratio': np.linspace(0, 1, 10),
        'logisticregression__C': np.logspace(-3, 2, 10)
    }

    pipeline_base = imblearn.pipeline.make_pipeline(StandardScaler(),
                                                    SMOTE(sampling_strategy=0.5, random_state=42),
                                                    LogisticRegression(random_state=42,
                                                                       penalty='elasticnet',
                                                                       solver='saga',
                                                                       max_iter=5000)
                                                    )

    gs = GridSearchCV(pipeline_base,
                      param_grid=param_grid,
                      cv=skf,
                      scoring='average_precision',
                      n_jobs=-1)

    gs.fit(x_train, y_train)
    print(f'Лучшие параметры: {gs.best_params_}')
    print(f'Лучшее качество на кросс-валидации: {gs.best_score_:.3f}')

    y_pred_proba = gs.predict_proba(x_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    print(f"ROC-AUC: {roc_auc:.4f}")

    y_pred = gs.predict(x_val)
    report = classification_report(y_val, y_pred)
    print("\nClassification Report:")
    print(report)

    best_pipe = gs.best_estimator_

    # Get step names
    logreg = best_pipe.named_steps['logisticregression']

    # coefficients (1D array)
    coefs = logreg.coef_[0]

    # get original feature names
    feature_names = x_train.columns

    # build dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'coef': coefs,
        'importance_abs': np.abs(coefs)
    }).sort_values('importance_abs', ascending=False)

    print(importance_df.head(20))

if __name__ == "__main__":
    baseline()