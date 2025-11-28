from pathlib import Path
import pandas as pd

def load_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    return df

def basic_checks(df):
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'na_percent': (df.isna().mean() * 100).round(3).to_dict()
    }
    return info
