"""
v19: D_minimal 기반 개별 피처 추가 실험
D_minimal에서 제거된 9개 피처를 하나씩 다시 추가
→ Public 0.79186에서 추가 개선 가능성 탐색

D_minimal 기준 (10개):
  NUM: Age, Fare, FamilySize
  CAT: Pclass, Sex, Title, Has_Cabin, Pclass_Sex
  OOF: SexSurname_k3, Ticket_k5

추가 후보:
  NUM: SibSp, Parch, TicketFreq, LogFare
  CAT: Embarked, IsAlone, Cabin_Deck, FamilySizeGroup, AgeGroup
"""
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_raw
from src.features import build_features
from src.config import RF_PARAMS, XGB_PARAMS, SUBMISSIONS

BASE_MODELS = {
    'LR':  LogisticRegression(C=1.0, max_iter=1000),
    'RF':  RandomForestClassifier(**RF_PARAMS),
    'GB':  GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1),
    'XGB': xgb.XGBClassifier(**XGB_PARAMS),
    'LGBM_LL': lgb.LGBMClassifier(
        n_estimators=569, num_leaves=31, learning_rate=0.010,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        n_jobs=-1, verbose=-1
    ),
}

SEEDS_QUICK = [42, 7, 21, 99, 123]
SEEDS_FULL  = [42, 7, 21, 99, 123, 0, 1, 2, 3, 4]

D_NUM = ["Age", "Fare", "FamilySize"]
D_CAT = ["Pclass", "Sex", "Title", "Has_Cabin", "Pclass_Sex"]

# 개별 추가 후보
ADDITIONS = {
    "base":          {"add_num": [],             "add_cat": []},
    "+SibSp":        {"add_num": ["SibSp"],       "add_cat": []},
    "+Parch":        {"add_num": ["Parch"],       "add_cat": []},
    "+TicketFreq":   {"add_num": ["TicketFreq"],  "add_cat": []},
    "+LogFare":      {"add_num": ["LogFare"],     "add_cat": []},
    "+Embarked":     {"add_num": [],             "add_cat": ["Embarked"]},
    "+IsAlone":      {"add_num": [],             "add_cat": ["IsAlone"]},
    "+Cabin_Deck":   {"add_num": [],             "add_cat": ["Cabin_Deck"]},
    "+FamSizeGrp":   {"add_num": [],             "add_cat": ["FamilySizeGroup"]},
    "+AgeGroup":     {"add_num": [],             "add_cat": ["AgeGroup"]},
    # 조합 후보 (5-seed에서 유망한 것들)
    "+SibSp+Parch":  {"add_num": ["SibSp", "Parch"], "add_cat": []},
}


def bayesian_smooth(counts, means, global_mean, k):
    return (counts * means + k * global_mean) / (counts + k)


def add_oof_bayesian(train_eng, y, test_eng, group_col, skf, k):
    global_mean = y.mean()
    oof_enc = np.full(len(train_eng), global_mean)
    groups_train = train_eng[group_col].values
    groups_test  = test_eng[group_col].values
    for tr_idx, val_idx in skf.split(train_eng, y):
        y_tr = y[tr_idx]
        g_tr = groups_train[tr_idx]
        counts   = pd.Series(g_tr).value_counts()
        means    = pd.Series(y_tr).groupby(g_tr).mean()
        smoothed = bayesian_smooth(counts, means, global_mean, k)
        oof_enc[val_idx] = pd.Series(groups_train[val_idx]).map(smoothed).fillna(global_mean).values
    counts_full   = pd.Series(groups_train).value_counts()
    means_full    = pd.Series(y).groupby(groups_train).mean()
    smoothed_full = bayesian_smooth(counts_full, means_full, global_mean, k)
    test_enc = pd.Series(groups_test).map(smoothed_full).fillna(global_mean).values
    return oof_enc, test_enc


def make_features(train_eng, y_train, test_eng, skf, num_feats, cat_feats):
    train_eng = train_eng.copy()
    test_eng  = test_eng.copy()

    # LogFare 계산 (필요 시)
    train_eng["LogFare"] = np.log1p(train_eng["Fare"].fillna(train_eng["Fare"].median()))
    test_eng["LogFare"]  = np.log1p(test_eng["Fare"].fillna(test_eng["Fare"].median()))

    train_eng["SexSurname"] = train_eng["Sex"] + "_" + train_eng["Surname"]
    test_eng["SexSurname"]  = test_eng["Sex"]  + "_" + test_eng["Surname"]

    ss_tr,   ss_te   = add_oof_bayesian(train_eng, y_train, test_eng, "SexSurname", skf, k=3)
    tick_tr, tick_te = add_oof_bayesian(train_eng, y_train, test_eng, "Ticket",     skf, k=5)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline,     num_feats),
        ("cat", categorical_pipeline, cat_feats),
    ], remainder="drop")

    X_tr_base = preprocessor.fit_transform(train_eng[num_feats + cat_feats])
    X_te_base = preprocessor.transform(test_eng[num_feats + cat_feats])

    X_train = np.hstack([X_tr_base, ss_tr.reshape(-1, 1), tick_tr.reshape(-1, 1)])
    X_test  = np.hstack([X_te_base, ss_te.reshape(-1, 1), tick_te.reshape(-1, 1)])
    return X_train, X_test


def stacking_cv(X_train, y_train, X_test, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    n_base = len(BASE_MODELS)
    oof_meta  = np.zeros((len(y_train), n_base))
    test_meta = np.zeros((len(X_test),  n_base))

    for tr_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr = y_train[tr_idx]
        fold_test_preds = []
        for i, (name, model) in enumerate(BASE_MODELS.items()):
            m = clone(model)
            try:
                m.set_params(random_state=seed)
            except ValueError:
                pass
            m.fit(X_tr, y_tr)
            oof_meta[val_idx, i] = m.predict_proba(X_val)[:, 1]
            fold_test_preds.append(m.predict_proba(X_test)[:, 1])
        test_meta += np.column_stack(fold_test_preds) / 5

    cv_scores = []
    for tr_idx, val_idx in skf.split(oof_meta, y_train):
        meta = LogisticRegression(C=0.05, max_iter=500)
        meta.fit(oof_meta[tr_idx], y_train[tr_idx])
        cv_scores.append((meta.predict(oof_meta[val_idx]) == y_train[val_idx]).mean())

    meta_final = LogisticRegression(C=0.05, max_iter=500)
    meta_final.fit(oof_meta, y_train)
    return np.array(cv_scores), meta_final.predict_proba(test_meta)[:, 1]


def run_seeds(label, num_feats, cat_feats, train_eng, y_train, test_eng, skf_base, seeds, baseline):
    total = len(num_feats) + len(cat_feats) + 2
    print(f"\n--- {label}  ({total}개 피처) ---")
    sys.stdout.flush()

    X_train, X_test = make_features(train_eng, y_train, test_eng, skf_base, num_feats, cat_feats)
    cv_means, probas = [], []
    for seed in seeds:
        scores, tp = stacking_cv(X_train, y_train, X_test, seed)
        cv_means.append(scores.mean())
        probas.append(tp)
        print(f"  seed={seed:3d} | CV={scores.mean():.4f}")
        sys.stdout.flush()

    avg  = np.mean(cv_means)
    std  = np.std(cv_means)
    diff = avg - baseline
    n_surv = (np.mean(probas, axis=0) >= 0.5).sum()
    star = " ★" if diff > 0.001 else (" △" if diff > 0 else "")
    print(f"  → {len(seeds)}-seed: {avg:.4f} ±{std:.4f}  생존:{n_surv}  diff={diff:+.4f}{star}")
    sys.stdout.flush()
    return avg, std, n_surv, np.mean(probas, axis=0)


def main():
    train, test = load_raw()
    _, y_train, _, test_ids, _, _, train_eng, test_eng = build_features(train, test)
    skf_base = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    BASELINE_CV = 0.8438  # D_minimal 10-seed

    print("=" * 60)
    print("v19: D_minimal 개별 피처 추가 (5-seed 빠른 탐색)")
    print(f"기준: D_minimal CV {BASELINE_CV:.4f}  Public 0.79186")
    print("=" * 60)

    results_5s = {}
    for label, spec in ADDITIONS.items():
        num_feats = D_NUM + spec["add_num"]
        cat_feats = D_CAT + spec["add_cat"]
        avg, std, ns, proba = run_seeds(
            label, num_feats, cat_feats,
            train_eng, y_train, test_eng, skf_base,
            SEEDS_QUICK, BASELINE_CV
        )
        results_5s[label] = (avg, std, ns, proba, num_feats, cat_feats)

    print(f"\n{'='*60}")
    print("5-seed 요약")
    print(f"{'실험':<16} | {'CV':>7} | {'std':>7} | {'생존':>5} | {'diff':>7}")
    print("-" * 60)
    base_cv = results_5s["base"][0]
    for label, (cv, std, ns, _, nf, cf) in results_5s.items():
        diff = cv - base_cv
        star = " ★" if diff > 0.001 else (" △" if diff > 0 else "")
        print(f"{label:<16} | {cv:.4f} | ±{std:.4f} | {ns:>5} | {diff:+.4f}{star}")
    sys.stdout.flush()

    # 10-seed 검증
    candidates = {l: r for l, r in results_5s.items()
                  if l != "base" and r[0] - base_cv > 0.001}

    if candidates:
        print(f"\n{'='*60}")
        print(f"10-seed 검증 ({len(candidates)}개 후보)")
        print("=" * 60)
        results_10s = {}
        for label, (_, _, _, _, num_feats, cat_feats) in candidates.items():
            avg, std, ns, proba = run_seeds(
                label + "_10s", num_feats, cat_feats,
                train_eng, y_train, test_eng, skf_base,
                SEEDS_FULL, BASELINE_CV
            )
            results_10s[label] = (avg, std, ns, proba)
            # 베이스라인보다 좋으면 저장
            if avg > BASELINE_CV:
                preds = (proba >= 0.5).astype(int)
                fname = f"submission_v19_{label.replace('+','p').replace(' ','_').lower()}_10s.csv"
                pd.DataFrame({"PassengerId": test_ids, "Survived": preds}).to_csv(
                    SUBMISSIONS / fname, index=False)
                print(f"  → 저장: {fname}")
                sys.stdout.flush()

        print(f"\n{'='*60}")
        print("10-seed 결과")
        print(f"{'실험':<18} | {'CV':>7} | {'std':>7} | {'생존':>5} | {'diff':>7}")
        print("-" * 60)
        for label, (cv, std, ns, _) in results_10s.items():
            diff = cv - BASELINE_CV
            star = " ★" if diff > 0 else ""
            print(f"{label:<18} | {cv:.4f} | ±{std:.4f} | {ns:>5} | {diff:+.4f}{star}")
        sys.stdout.flush()
    else:
        print("\n10-seed 스킵: 5-seed에서 유의미한 개선 없음.")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
