import numpy as np
import pandas as pd
import json, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_NAMES = json.load(open(os.path.join(BASE, 'model', 'feature_names.json'), 'r', encoding='utf-8'))
DUP_COLS = json.load(open(os.path.join(BASE, 'model', 'dup_cols.json'), 'r', encoding='utf-8'))


def preprocess_input(df: pd.DataFrame):
    if df.shape[1] == len(FEATURE_NAMES) + 1:
        X_df = df.iloc[:, :-1]
    elif df.shape[1] == len(FEATURE_NAMES):
        X_df = df.copy()
    else:
        raise ValueError(f"列数不匹配。期望 {len(FEATURE_NAMES)} 个特征列，实际 {df.shape[1]} 列。")

    if len(X_df.columns) == len(FEATURE_NAMES):
        X_df.columns = FEATURE_NAMES
    else:
        X_df = X_df.reindex(columns=FEATURE_NAMES, fill_value=0.0)

    X_df = X_df.drop(columns=[c for c in DUP_COLS if c in X_df.columns])
    X_df = X_df[FEATURE_NAMES]
    X = np.nan_to_num(X_df.values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    return X, X_df.index.tolist()