"""
v21: DC_F 기반 심화 실험
DC_F = Dmin + CabinDeck + FamilySizeGroup → 0.79665 신기록

다음 시도:
A: DC_F + SVM
B: DC_F + IsAlone
C: DC_F + AgeGroup
D: DC_F + SibSp
E: DC_F blended with DC (138surv)
F: DC_F blended with DF (136surv)
G: DC_F blended with blend60_Dmin_v10 (136surv)
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
from sklearn.svm import SVC
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

BASE_5 = {
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
BASE_6 = {**BASE_5, 'SVM': SVC(C=1.0, kernel='rbf', probability=True)}

SEEDS_FULL = [42, 7, 21, 99, 123, 0, 1, 2, 3, 4]
BASELINE = 0.8448  # DC_F 10-seed CV

D_NUM = ["Age", "Fare", "FamilySize"]
DCF_CAT = ["Pclass", "Sex", "Title", "Has_Cabin", "Pclass_Sex", "Cabin_Deck", "FamilySizeGroup"]


def bayesian_smooth(c, m, g, k): return (c * m + k * g) / (c + k)


def add_oof_bayesian(train_eng, y, test_eng, group_col, skf, k):
    g_mean = y.mean()
    oof = np.full(len(train_eng), g_mean)
    gtr = train_eng[group_col].values
    gte = test_eng[group_col].values
    for tri, vi in skf.split(train_eng, y):
        ytr_, gtr_ = y[tri], gtr[tri]
        cnt = pd.Series(gtr_).value_counts()
        mn  = pd.Series(ytr_).groupby(gtr_).mean()
        sm  = bayesian_smooth(cnt, mn, g_mean, k)
        oof[vi] = pd.Series(gtr[vi]).map(sm).fillna(g_mean).values
    cnt_f = pd.Series(gtr).value_counts()
    mn_f  = pd.Series(y).groupby(gtr).mean()
    sm_f  = bayesian_smooth(cnt_f, mn_f, g_mean, k)
    return oof, pd.Series(gte).map(sm_f).fillna(g_mean).values


def make_features(train_eng, y_train, test_eng, skf, num_f, cat_f):
    tr = train_eng.copy(); te = test_eng.copy()
    tr["LogFare"] = np.log1p(tr["Fare"].fillna(tr["Fare"].median()))
    te["LogFare"] = np.log1p(te["Fare"].fillna(te["Fare"].median()))
    tr["SexSurname"] = tr["Sex"] + "_" + tr["Surname"]
    te["SexSurname"] = te["Sex"] + "_" + te["Surname"]
    ss_tr, ss_te = add_oof_bayesian(tr, y_train, te, "SexSurname", skf, 3)
    tk_tr, tk_te = add_oof_bayesian(tr, y_train, te, "Ticket",     skf, 5)
    num_p = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())])
    cat_p = Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                      ("e", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
    pre = ColumnTransformer([("n", num_p, num_f), ("c", cat_p, cat_f)], remainder="drop")
    Xtr = np.hstack([pre.fit_transform(tr[num_f + cat_f]),
                     ss_tr.reshape(-1, 1), tk_tr.reshape(-1, 1)])
    Xte = np.hstack([pre.transform(te[num_f + cat_f]),
                     ss_te.reshape(-1, 1), tk_te.reshape(-1, 1)])
    return Xtr, Xte


def stacking_cv(Xtr, ytr, Xte, seed, base_models):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros((len(ytr), len(base_models)))
    tst = np.zeros((len(Xte),  len(base_models)))
    for tri, vi in skf.split(Xtr, ytr):
        fp = []
        for i, (nm, m) in enumerate(base_models.items()):
            mc = clone(m)
            try: mc.set_params(random_state=seed)
            except ValueError: pass
            mc.fit(Xtr[tri], ytr[tri])
            oof[vi, i] = mc.predict_proba(Xtr[vi])[:, 1]
            fp.append(mc.predict_proba(Xte)[:, 1])
        tst += np.column_stack(fp) / 5
    cvs = []
    for tri, vi in skf.split(oof, ytr):
        meta = LogisticRegression(C=0.05, max_iter=500)
        meta.fit(oof[tri], ytr[tri])
        cvs.append((meta.predict(oof[vi]) == ytr[vi]).mean())
    mf = LogisticRegression(C=0.05, max_iter=500)
    mf.fit(oof, ytr)
    return np.array(cvs), mf.predict_proba(tst)[:, 1]


def run_exp(label, num_f, cat_f, tr_eng, ytr, te_eng, skf, seeds, base_models=None):
    if base_models is None: base_models = BASE_5
    print(f"\n--- {label}  ({len(num_f)+len(cat_f)+2}feat) ---")
    sys.stdout.flush()
    Xtr, Xte = make_features(tr_eng, ytr, te_eng, skf, num_f, cat_f)
    cvs, probs = [], []
    for seed in seeds:
        sc, tp = stacking_cv(Xtr, ytr, Xte, seed, base_models)
        cvs.append(sc.mean()); probs.append(tp)
        print(f"  seed={seed:3d} | CV={sc.mean():.4f}")
        sys.stdout.flush()
    avg = np.mean(cvs); std = np.std(cvs)
    proba = np.mean(probs, axis=0)
    ns = (proba >= 0.5).sum()
    diff = avg - BASELINE
    star = " ★" if diff > 0.001 else (" △" if diff > 0 else "")
    print(f"  → {len(seeds)}-seed: {avg:.4f} ±{std:.4f}  생존:{ns}  diff={diff:+.4f}{star}")
    sys.stdout.flush()
    return avg, std, ns, proba


def main():
    train, test = load_raw()
    _, ytr, _, test_ids, _, _, tr_eng, te_eng = build_features(train, test)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("=" * 65)
    print("v21: DC_F 기반 심화 실험 (10-seed)")
    print(f"기준: DC_F 0.8448  Public 0.79665")
    print("=" * 65)

    experiments = {
        "DCF_base": (D_NUM, DCF_CAT,                BASE_5),
        "DCF_SVM":  (D_NUM, DCF_CAT,                BASE_6),
        "DCF+IsAlone": (D_NUM, DCF_CAT+["IsAlone"], BASE_5),
        "DCF+AgeGrp":  (D_NUM, DCF_CAT+["AgeGroup"],BASE_5),
        "DCF+SibSp":   (D_NUM+["SibSp"], DCF_CAT,   BASE_5),
        "DCF+Embarked":(D_NUM, DCF_CAT+["Embarked"],BASE_5),
    }

    results = {}
    for label, (nf, cf, bm) in experiments.items():
        avg, std, ns, proba = run_exp(label, nf, cf, tr_eng, ytr, te_eng, skf, SEEDS_FULL, bm)
        results[label] = (avg, std, ns, proba)

    print(f"\n{'='*65}")
    print("10-seed 결과 요약")
    print(f"{'실험':<16} | {'CV':>7} | {'std':>7} | {'생존':>5} | {'diff':>7}")
    print("-" * 65)
    for lb, (cv, std, ns, _) in results.items():
        diff = cv - BASELINE
        star = " ★" if diff > 0.001 else (" △" if diff > 0 else "")
        print(f"{lb:<16} | {cv:.4f} | ±{std:.4f} | {ns:>5} | {diff:+.4f}{star}")
    sys.stdout.flush()

    # 모든 파일 저장
    print(f"\n{'='*65}")
    print("제출 파일 생성")
    dcf_p = results["DCF_base"][3]
    for lb, (_, _, ns, proba) in results.items():
        preds = (proba >= 0.5).astype(int)
        fname = f"submission_v21_{lb.lower().replace('+','_').replace(' ','')}_10s.csv"
        pd.DataFrame({"PassengerId": test_ids, "Survived": preds}).to_csv(SUBMISSIONS / fname, index=False)
        print(f"  저장: {fname}  생존:{ns}")
        sys.stdout.flush()

    # DCF 기반 블렌딩 (다양한 비율 DC_F×DC_SVM)
    dcf_svm_p = results["DCF_SVM"][3]
    for w in [0.60, 0.70, 0.80]:
        blended = w * dcf_p + (1-w) * dcf_svm_p
        preds = (blended >= 0.5).astype(int)
        ns = preds.sum()
        fname = f"submission_v21_blend_dcf{int(w*100)}_dcfsvm{int((1-w)*100)}.csv"
        pd.DataFrame({"PassengerId": test_ids, "Survived": preds}).to_csv(SUBMISSIONS / fname, index=False)
        print(f"  저장: {fname}  생존:{ns}")
        sys.stdout.flush()

    print("\n완료.")


if __name__ == "__main__":
    main()
