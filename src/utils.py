# src/utils.py
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set(style="whitegrid")

# -----------------------
# I/O helpers
# -----------------------
def ensure_parent_dir(path):
    """Ensure parent directory for a file path exists."""
    parent = os.path.dirname(str(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

def save_json(obj, path):
    """Save object as JSON (creates parent dir if needed)."""
    ensure_parent_dir(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
    print(f"Saved JSON to: {path}")

def save_model(model, path):
    """Save sklearn-compatible model using joblib."""
    ensure_parent_dir(path)
    joblib.dump(model, path)
    print(f"Saved model to: {path}")

def load_model(path):
    """Load model from joblib file."""
    return joblib.load(path)

# -----------------------
# Plot helpers
# -----------------------
def plot_actual_vs_pred(y_true, y_pred, save_path=None, title="Actual vs Predicted"):
    """
    Scatter plot actual vs predicted + identity line + metrics in title.
    If save_path provided, saves PNG and closes figure.
    """
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lim_min = min(y_true.min(), y_pred.min())
    lim_max = max(y_true.max(), y_pred.max())
    plt.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{title}\nR2={r2:.3f}  MAE={mae:.1f}  RMSE={rmse:.1f}")
    plt.tight_layout()

    if save_path:
        ensure_parent_dir(save_path)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved figure to: {save_path}")
        return str(save_path)
    return plt

def plot_model_comparison(results_dict, metric='r2_test', save_path=None, title="Model comparison"):
    """
    Bar chart comparing models using a metric (e.g., 'r2_test' or 'mae_test').
    results_dict: {'ModelName': {'r2_test':..., 'mae_test':..., ...}, ...}
    """
    rows = []
    for name, res in results_dict.items():
        # if nested cv report, skip or extract relevant key
        if isinstance(res, dict):
            val = res.get(metric, None)
        else:
            val = None
        rows.append({'model': name, 'metric': val})
    df = pd.DataFrame(rows).dropna(subset=['metric'])

    if df.empty:
        print("No data to plot for metric:", metric)
        return None

    plt.figure(figsize=(6, max(2, 0.6 * len(df))))
    sns.barplot(x='metric', y='model', data=df.sort_values('metric', ascending=False), orient='h')
    plt.xlabel(metric)
    plt.ylabel('Model')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        ensure_parent_dir(save_path)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved figure to: {save_path}")
        return str(save_path)
    return plt

def plot_cv_scores(scores, title='CV R2 per fold', save_path=None):
    """Plot cross-validation fold scores (list-like)."""
    plt.figure(figsize=(6,3))
    sns.barplot(x=list(range(1, len(scores)+1)), y=scores)
    plt.xlabel('Fold')
    plt.ylabel('R2')
    plt.title(title)
    plt.ylim(0,1)
    plt.tight_layout()

    if save_path:
        ensure_parent_dir(save_path)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved CV plot to: {save_path}")
        return str(save_path)
    return plt
