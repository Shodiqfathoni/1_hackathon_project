import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

def evaluate_model(pipe, X_train, y_train, X_test, y_test):
    pipe.fit(X_train, y_train)
    y_pred_tr = pipe.predict(X_train)
    y_pred_te = pipe.predict(X_test)
    results = {
        'r2_train': float(r2_score(y_train, y_pred_tr)),
        'r2_test': float(r2_score(y_test, y_pred_te)),
        'mae_train': float(mean_absolute_error(y_train, y_pred_tr)),
        'mae_test': float(mean_absolute_error(y_test, y_pred_te)),
        'rmse_train': float(np.sqrt(mean_squared_error(y_train, y_pred_tr))),
        'rmse_test': float(np.sqrt(mean_squared_error(y_test, y_pred_te)))
    }
    return results

def cross_val_report(pipe, X, y, n_splits=5, scoring='r2'):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    report = {
        'scores': scores.tolist(),
        'mean': float(scores.mean()),
        'std': float(scores.std())
    }
    return report
