"""
Inference and submission generation.
"""
import pandas as pd
import numpy as np
from src.config import SUBMISSIONS


def generate_submission(preds: np.ndarray, test_ids: np.ndarray, filename: str) -> pd.DataFrame:
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"PassengerId": test_ids, "Survived": preds.astype(int)})
    path = SUBMISSIONS / filename
    sub.to_csv(path, index=False)
    print(f"Saved: {path.name} | Predicted survival: {preds.mean():.1%}")
    return sub
