from pathlib import Path

ROOT = Path(__file__).parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
SUBMISSIONS = ROOT / "data" / "submissions"
FIGURES_DIR = ROOT / "outputs" / "figures"
MODELS_DIR = ROOT / "outputs" / "models"
EDA_REPORT_DIR = ROOT / "eda_report"

RANDOM_SEED = 42
CV_FOLDS = 5
TARGET = "Survived"

NUM_FEATURES = [
    "Age", "Fare", "SibSp", "Parch", "FamilySize", "TicketFreq", "LogFare",
]
CAT_FEATURES = [
    "Pclass", "Sex", "Embarked", "Title", "IsAlone", "Has_Cabin",
    "Cabin_Deck", "FamilySizeGroup", "AgeGroup", "Pclass_Sex",
]

XGB_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=RANDOM_SEED, n_jobs=-1,
)

LGBM_PARAMS = dict(
    n_estimators=500, num_leaves=31, learning_rate=0.05,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
)

RF_PARAMS = dict(
    n_estimators=300, max_depth=8, min_samples_split=10,
    random_state=RANDOM_SEED, n_jobs=-1,
)
