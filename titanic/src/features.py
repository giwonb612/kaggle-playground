"""
Feature engineering pipeline for the Titanic dataset.
All transformations are fit on train only, then applied to test.
"""
import re
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from src.config import NUM_FEATURES, CAT_FEATURES, TARGET


# ── Domain transforms ────────────────────────────────────────────────────────

def _extract_title(name: str) -> str:
    # Robust regex to handle edge cases like "the Countess."
    m = re.search(r',\s*(.+?)\.', name)
    title = m.group(1).strip() if m else ""
    replacements = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    rare = {"Lady", "Countess", "the Countess", "Capt", "Col", "Don", "Dr",
            "Major", "Rev", "Sir", "Jonkheer", "Dona"}
    title = replacements.get(title, title)
    return "Rare" if title in rare else title


def _impute_age(df: pd.DataFrame) -> pd.Series:
    """Impute Age using median per (Pclass, Title) group."""
    age = df["Age"].copy()
    df_tmp = df.copy()
    df_tmp["Title"] = df_tmp["Name"].apply(_extract_title)
    medians = df_tmp.groupby(["Pclass", "Title"])["Age"].median()
    for idx in age[age.isna()].index:
        key = (df_tmp.loc[idx, "Pclass"], df_tmp.loc[idx, "Title"])
        if key in medians:
            age.loc[idx] = medians[key]
        else:
            age.loc[idx] = df_tmp.groupby("Pclass")["Age"].median().get(
                df_tmp.loc[idx, "Pclass"], df["Age"].median()
            )
    return age


def _family_size_group(fs: int) -> str:
    if fs == 1:
        return "Alone"
    elif fs <= 4:
        return "Small"
    else:
        return "Large"


def _age_group(age: float) -> str:
    if age <= 5:
        return "Baby"
    elif age <= 12:
        return "Child"
    elif age <= 17:
        return "Teen"
    elif age <= 60:
        return "Adult"
    else:
        return "Senior"


def engineer_features(df: pd.DataFrame,
                       ticket_counts: pd.Series = None,
                       surname_counts: pd.Series = None) -> pd.DataFrame:
    """
    Apply all domain-specific feature engineering.
    ticket_counts / surname_counts: pre-computed on combined train+test to
    avoid train/test frequency mismatch (pass from build_features).
    """
    out = df.copy()

    # Titles
    out["Title"] = out["Name"].apply(_extract_title)

    # Surname
    out["Surname"] = out["Name"].str.split(",").str[0].str.strip()

    # Age imputation
    out["Age"] = _impute_age(out)

    # Fare: fill single missing in test
    out["Fare"] = out["Fare"].fillna(
        out.groupby("Pclass")["Fare"].transform("median")
    )

    # Log fare
    out["LogFare"] = np.log1p(out["Fare"])

    # Embarked
    out["Embarked"] = out["Embarked"].fillna("S")

    # Family features
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["FamilySizeGroup"] = out["FamilySize"].apply(_family_size_group)

    # Cabin
    out["Has_Cabin"] = out["Cabin"].notna().astype(int)
    out["Cabin_Deck"] = out["Cabin"].str[0].fillna("Unknown")

    # Age group
    out["AgeGroup"] = out["Age"].apply(_age_group)

    # Interaction
    out["Pclass_Sex"] = out["Pclass"].astype(str) + "_" + out["Sex"]

    # Ticket frequency — use combined counts to avoid train/test mismatch
    if ticket_counts is not None:
        out["TicketFreq"] = out["Ticket"].map(ticket_counts).fillna(1)
    else:
        out["TicketFreq"] = out["Ticket"].map(out["Ticket"].value_counts())

    # Fare per person (shared ticket fares are split)
    out["FarePerPerson"] = out["Fare"] / out["TicketFreq"].replace(0, 1)
    out["LogFarePerPerson"] = np.log1p(out["FarePerPerson"])

    # IsChild: stronger signal than AgeGroup for "children first"
    out["IsChild"] = (out["Age"] <= 15).astype(int)

    # IsMother: married women with children — highest priority in evacuation
    out["IsMother"] = (
        (out["Sex"] == "female") &
        (out["Parch"] > 0) &
        (out["Age"] > 18) &
        (out["Title"] == "Mrs")
    ).astype(int)

    # Surname frequency — combined counts to avoid mismatch
    if surname_counts is not None:
        out["SurnameFreq"] = out["Surname"].map(surname_counts).fillna(1)
    else:
        out["SurnameFreq"] = out["Surname"].map(out["Surname"].value_counts())

    return out


# ── OOF survival encoding ────────────────────────────────────────────────────

def add_oof_survival_encoding(train_eng: pd.DataFrame, y: np.ndarray,
                               test_eng: pd.DataFrame, group_col: str, skf):
    """
    OOF mean target encoding for a group column.
    Train: K-Fold OOF (leak-free). Test: full train statistics.
    Returns (oof_enc array for train, enc array for test).
    """
    global_mean = y.mean()
    oof_enc = np.full(len(train_eng), global_mean)
    groups_train = train_eng[group_col].values
    groups_test = test_eng[group_col].values

    for tr_idx, val_idx in skf.split(train_eng, y):
        group_map = (
            pd.Series(y[tr_idx], index=tr_idx)
            .groupby(groups_train[tr_idx])
            .mean()
        )
        oof_enc[val_idx] = pd.Series(groups_train[val_idx]).map(group_map).fillna(global_mean).values

    full_map = pd.Series(y).groupby(groups_train).mean()
    test_enc = pd.Series(groups_test).map(full_map).fillna(global_mean).values
    return oof_enc, test_enc


# ── sklearn pipeline ─────────────────────────────────────────────────────────

def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUM_FEATURES),
        ("cat", categorical_pipeline, CAT_FEATURES),
    ], remainder="drop")

    return preprocessor


def build_features(train: pd.DataFrame, test: pd.DataFrame):
    """
    Full pipeline: engineer -> extract X/y -> fit preprocessor on train.
    Ticket/surname counts computed on combined train+test to avoid mismatch.
    Returns (X_train, y_train, X_test, test_ids, preprocessor, feature_names).
    """
    # Compute frequencies on combined data — prevents distribution mismatch
    all_data = pd.concat([train, test], sort=False)
    combined_ticket_counts = all_data["Ticket"].value_counts()
    # Surname: extract first
    all_data["_surname"] = all_data["Name"].str.split(",").str[0].str.strip()
    combined_surname_counts = all_data["_surname"].value_counts()

    train_eng = engineer_features(train,
                                   ticket_counts=combined_ticket_counts,
                                   surname_counts=combined_surname_counts)
    test_eng = engineer_features(test,
                                  ticket_counts=combined_ticket_counts,
                                  surname_counts=combined_surname_counts)

    y_train = train_eng[TARGET].values
    test_ids = test_eng["PassengerId"].values

    all_features = NUM_FEATURES + CAT_FEATURES
    X_train_raw = train_eng[all_features]
    X_test_raw = test_eng[all_features]

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    cat_names = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(CAT_FEATURES)
    feature_names = NUM_FEATURES + list(cat_names)

    return X_train, y_train, X_test, test_ids, preprocessor, feature_names, train_eng, test_eng
