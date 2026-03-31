"""
v22: DC_F sweet spot 주변 탐색
DC_F = Dmin + CabinDeck + FamilySizeGroup → 0.79665 신기록

다음 시도:
A: DC_F - Has_Cabin (Cabin_Deck이 has_cabin 정보 포함 → 중복 제거)
B: DC_F + Parch (SibSp 실패, Parch는 미시도)
C: DC_F + LogFare
D: DC_F + TicketFreq
E: DC_F - Has_Cabin + Parch (A+B 조합)
F: DC_F + CatBoost (베이스 모델 다양성)
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

try:
    from catboost import CatBoostClassifier
    BASE_5_CB = {
        **BASE_5,
        'CB': CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            loss_function='Logloss', verbose=False, random_seed=42
        )
    }
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost 없음 — F 실험 스킵")

SEEDS_FULL = [42, 7, 21, 99, 123, 0, 1, 2, 3, 4]
BASELINE = 0.8448  # DC_F 10-seed CV

D_NUM = ["Age", "Fare", "FamilySize"]
DCF_CAT     = ["Pclass", "Sex", "Title", "Has_Cabin", "Pclass_Sex", "Cabin_Deck", "FamilySizeGroup"]
DCF_noCabin = ["Pclass", "Sex", "Title",              "Pclass_Sex", "Cabin_Deck", "FamilySizeGroup"]  # A


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
    print("v22: DC_F 주변 탐색 (10-seed)")
    print(f"기준: DC_F 0.8448  Public 0.79665")
    print("=" * 65)

    experiments = {
        "DCF_base":      (D_NUM,            DCF_CAT,                     BASE_5),
        "DCF_noCabin":   (D_NUM,            DCF_noCabin,                 BASE_5),  # A: Has_Cabin 제거
        "DCF+Parch":     (D_NUM+["Parch"],  DCF_CAT,                     BASE_5),  # B
        "DCF+LogFare":   (D_NUM,            DCF_CAT,                     BASE_5),  # C (LogFare는 make_features에서 추가됨 — 수동 추가)
        "DCF+TicketFreq":(D_NUM+["TicketFreq"], DCF_CAT,                 BASE_5),  # D
        "DCF_nC+Parch":  (D_NUM+["Parch"],  DCF_noCabin,                 BASE_5),  # E: A+B
    }

    # C: DCF+LogFare — LogFare를 명시적으로 num에 추가
    # make_features에서 LogFare를 항상 계산하지만 num_f에 넣어야 포함됨
    experiments["DCF+LogFare"] = (D_NUM + ["LogFare"], DCF_CAT, BASE_5)

    if HAS_CATBOOST:
        experiments["DCF_CB"] = (D_NUM, DCF_CAT, BASE_5_CB)  # F

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

    # 제출 파일 저장 (CV >= BASELINE인 것만)
    print(f"\n{'='*65}")
    print("제출 파일 생성 (baseline 이상)")
    dcf_base_p = results["DCF_base"][3]
    saved = []
    for lb, (cv, std, ns, proba) in results.items():
        if lb == "DCF_base":
            continue  # 이미 제출됨
        preds = (proba >= 0.5).astype(int)
        fname = f"submission_v22_{lb.lower().replace('+','_').replace(' ','')}_10s.csv"
        pd.DataFrame({"PassengerId": test_ids, "Survived": preds}).to_csv(SUBMISSIONS / fname, index=False)
        saved.append((fname, ns, cv - BASELINE))
        print(f"  저장: {fname}  생존:{ns}  diff={cv-BASELINE:+.4f}")
        sys.stdout.flush()

    # 최고 성과 파일과 DCF_base 블렌딩
    best_label = max([(lb, cv) for lb, (cv, *_) in results.items() if lb != "DCF_base"], key=lambda x: x[1])[0]
    best_p = results[best_label][3]
    for w in [0.60, 0.70, 0.80]:
        blended = w * dcf_base_p + (1-w) * best_p
        preds = (blended >= 0.5).astype(int)
        ns = preds.sum()
        fname = f"submission_v22_blend_dcf{int(w*100)}_{best_label.lower()}{int((1-w)*100)}.csv"
        pd.DataFrame({"PassengerId": test_ids, "Survived": preds}).to_csv(SUBMISSIONS / fname, index=False)
        print(f"  저장: {fname}  생존:{ns}")
        sys.stdout.flush()

    print("\n완료.")


if __name__ == "__main__":
    main()
