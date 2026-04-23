import joblib, json, os
import numpy as np
from typing import Tuple, List

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE = joblib.load(os.path.join(BASE, 'model', 'model.pkl'))
CLASS_INFO = json.load(open(os.path.join(BASE, 'model', 'class_labels.json'), 'r', encoding='utf-8'))

def predict(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    proba = PIPELINE.predict_proba(X)
    pred = np.argmax(proba, axis=1)
    return pred, proba, CLASS_INFO['class_names']