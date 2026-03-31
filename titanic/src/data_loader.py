import pandas as pd
from src.config import DATA_RAW


def load_raw():
    train = pd.read_csv(DATA_RAW / "train.csv")
    test = pd.read_csv(DATA_RAW / "test.csv")
    return train, test


def load_gender_submission():
    return pd.read_csv(DATA_RAW / "gender_submission.csv")
