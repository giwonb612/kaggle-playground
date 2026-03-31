# Titanic ML — 실험 기록

> 마지막 업데이트: 2026-03-25

---

## 현재 최고 점수: **0.79665** (v20 DC_F 10-seed)

---

## 실험 이력

### v1 — 초기 baseline
- 피처: 기본 (Title, FamilySize, Has_Cabin)
- 모델: Stacking (LR, RF, GB, XGB, LGBM, SVM)
- Public: 0.76555

### v2 — TicketFreq 수정 시도 (실패)
- 변경: IsChild, IsMother, SurnameFreq 추가
- 결과: Public 하락 (소규모 데이터 노이즈)
- **교훈**: 891개 데이터에서 새 피처 추가 = 노이즈 위험

### v3 — TicketFreq 버그 수정
- 변경: train/test 개별 계산 → combined 계산
- 결과: +0.005 향상

### v4 — 피처셋 정리 (**0.77511**)
- 피처: NUM 7개 + CAT 10개 = 17개
- 모델: Stacking v4
- **이 피처셋이 현재 기준**

### v5 — Optuna 튜닝 (**0.76276**)
- 변경: Optuna 60 trials
- 결과: -0.013 하락
- **교훈**: CV 자체에 과적합. 소규모 데이터에서 강한 튜닝은 역효과

### v6 — FamilySurvivalRate 누수 버전 (**0.76555**)
- 변경: Surname+Ticket 그룹 생존율 피처 추가 (전체 훈련으로 계산)
- CV: 88.55% (과대평가)
- **교훈**: 전체 훈련 기반 그룹 통계 = 누수 → CV 부풀림 → Public 하락

### v7 — OOF 생존율 인코딩 (**0.77751**)
- 변경: Surname + Ticket OOF 생존율 (K-Fold OOF, 누수 없음)
- CV: 85.18%
- **교훈**: 올바른 OOF 방식이 핵심. CV와 Public의 일치도 상승

### v8 실험들
| 파일 | Public | CV | 비고 |
|---|---|---|---|
| v8_oof_stacking.csv | 0.77511 | 83.50% | raw OOF (미스무딩) |
| v8_lgbm_lowlr.csv | 0.77511 | 84.51% | LGBM solo → stacking보다 못함 |
| v8_multiseed.csv | 0.76076 | 82.94% | multi-seed solo → 더 나쁨 |
| v8_combined.csv | 0.77033 | - | 블렌딩 → 중간 |

**v8 분석**: raw OOF(스무딩 없음)는 n=1 성씨의 생존율 0%가 3등석 여성을 오분류

### v9 실험들
| 파일 | Public | CV | 비고 |
|---|---|---|---|
| **v9_stacking.csv** | **0.77990** | 83.88% | Sex+Surname OOF k=3 + Ticket k=5 ★ |
| v9_v7_or.csv | 0.76315 | - | v7 OR v9 블렌딩 → 노이즈 추가 |
| v9_lgbm_multiseed.csv | 0.76794 | 84.40% | LGBM solo → stacking보다 못함 |

**v9 핵심 발견**:
- `Sex+Surname` 그룹 분리가 핵심 (Olsen 남성 사망 ≠ Olsen 여성 운명)
- Bayesian Smoothing (k=3~5)으로 n=1 케이스 완화
- **Stacking이 단일 모델보다 항상 우수** (v8, v9 공통)

---

## 핵심 교훈 요약

| 실험 | 효과 | 이유 |
|---|---|---|
| TicketFreq combined | +0.005 ✅ | train/test 분포 일치 |
| 새 피처 (IsChild 등) | -0.007 ❌ | 891개 → 노이즈 |
| Optuna 60 trials | -0.013 ❌ | CV 과적합 |
| OOF 생존율 (v7) | +0.002 ✅ | 누수 없는 정보 |
| Sex+Surname OOF (v9) | +0.002 ✅ | 성별 분리로 오분류 방지 |
| LGBM solo (any) | -0.002~0.015 ❌ | Stacking이 항상 우수 |
| OR 블렌딩 | -0.017 ❌ | 노이즈 추가 |

---

## CV vs Public Score 패턴

```
Stacking meta-CV ≈ Public × 1.05~1.07
LGBM/XGB solo CV ≈ Public × 1.08~1.10

신뢰할 수 있는 CV: Stacking meta-CV, 분산 ±1% 이하
신뢰 낮은 CV: solo 모델 CV, Optuna 최적화 후 CV
```

---

## v10 실험 결과 (2026-03-21)

### 메타러너 C값 변경 — 단일 seed vs 10-seed 결과 불일치
| 설정 | 단일 seed(42) CV | 10-seed 평균 CV | 판정 |
|---|---|---|---|
| 기준 LR C=1.0 | 83.61% | 84.23% | 기준 |
| LR C=0.05 | 84.40% (+0.79%p) ★ | 84.14% (-0.09%p) | ❌ 노이즈 |
| LR C=0.1 | 84.29% (+0.68%p) ★ | 84.22% (-0.01%p) | ❌ 노이즈 |

**교훈**: 단일 seed CV 차이는 노이즈일 수 있음. **반드시 10-seed 반복으로 확인 후 판단.**

### Stacking 구조 변경 — 10-seed 결과 (SVM 포함 6 base)
| 설정 | 10-seed CV | 분산 | diff |
|---|---|---|---|
| **LGBM→LowLR + C=0.05** | **84.12%** | ±0.35% | **+0.24%p** ★ |
| -SVM + C=0.05 | 84.09% | ±0.59% | +0.21%p |
| C=0.1 | 84.04% | ±0.38% | +0.16%p |
| C=0.05 | 84.03% | ±0.40% | +0.15%p |
| **기준 C=1.0** | 83.88% | ±0.31% | 기준 (Public 0.77990) |

**핵심**: LGBM 베이스 모델을 low-LR(lr=0.01, n=569)로 교체 + meta C=0.05 → +0.24%p 일관 개선.
단일 모델 LGBM solo는 Public에서 약했지만, **베이스 모델로 쓰는 것은 다름** — 스태킹 다양성 기여.
→ **제출 결과: Public 0.78708 ★ 신기록** (+0.00718 vs v9)

## v11 실험 결과 (2026-03-22)

### OOF 피처 다양화 — 모두 실패
| 설정 | 5-seed CV | diff | 비고 |
|---|---|---|---|
| v10_nosvm baseline | 84.20% | 기준 | SVM 없어도 유사 |
| A: TicketPrefix OOF | 83.61% | -0.58%p ❌ | 그룹 너무 coarse |
| B: AgeSex OOF | 83.35% | -0.85%p ❌ | 기존 피처와 중복 |
| C: 둘 다 추가 | 83.37% | -0.83%p ❌ | 복합 노이즈 |

**교훈**: TicketPrefix (PC/CA/OTHER 등)는 너무 굵은 분류. AgeSex는 AgeGroup+Sex 조합이지만 기존 피처와 상관관계 높아 OOF로 추가 시 오히려 노이즈. 소규모 데이터에서 그룹 OOF의 효과는 그룹이 **의미 있는 생존 차별성**을 가질 때만 유효 (Sex+Surname이 그 예).

## v12 실험 결과 (2026-03-22)

### k값 그리드 서치 — 전범위 탐색, 개선 없음
30개 조합 (k_ss=[1,2,3,5,7,10] × k_tick=[1,3,5,7,10]) 탐색 결과:

| 발견 | 내용 |
|---|---|
| Phase1 공동 1위 | k_ss=5,7 / k_tick=7: +0.001 (5-seed) |
| Phase2 검증 | 모두 -0.001~-0.002 (10-seed) → 노이즈 |
| 패턴 | k_tick=7~8 구간이 최적, k_tick≥10은 항상 하락 |
| 결론 | v10 k_ss=3, k_tick=5가 이미 최적에 가까움 |

**교훈**: 5-seed에서 +0.001 개선도 10-seed에서 노이즈로 반전될 수 있음. k값 그리드 서치는 소규모 데이터에서 효과 제한적.

## v13 실험 결과 (2026-03-23)

### 피처 삭제 실험 — 핵심 발견: 피처 줄이면 개선!

| 설정 | 10-seed CV | diff | 생존수 |
|---|---|---|---|
| v10_full (19개) | 84.12% | 기준 | 133명 |
| B_noIslands (17개, Embarked/IsAlone 제거) | 84.30% | +0.18%p ★ | 138명 |
| C_core12 (14개) | 84.34% | +0.22%p ★ | 138명 |
| **D_minimal (10개)** | **84.38%** | **+0.26%p ★** | 137명 |

**핵심 교훈**: 피처 수 줄이면 개선 (단조적 패턴). 소규모 데이터(891개)에서 19개 피처 중 다수가 노이즈 → 제거하면 일반화 향상.

D_minimal 피처:
```python
NUM = ["Age", "Fare", "FamilySize"]
CAT = ["Pclass", "Sex", "Title", "Has_Cabin", "Pclass_Sex"]
OOF = [SexSurname_k3, Ticket_k5]  # 총 10개
```

## v13 Public Score 결과 (2026-03-23) ⭐ 신기록

| 설정 | CV | Public | diff vs v10 |
|---|---|---|---|
| v10_full (19개) | 84.12% | 0.78708 | 기준 |
| B_noIslands (17개) | 84.30% | (pending) | - |
| C_core12 (14개) | 84.34% | **0.78947** | +0.00239 ★ |
| **D_minimal (10개)** | **84.38%** | **0.79186** | **+0.00478 ★★** |

**핵심 교훈**:
- 피처 축소(19→10)가 Public +0.00478 개선 (신기록!)
- FamilySize는 테스트셋에서 필수 피처 → 제거 시 0.78708로 하락
- CV 개선 ≠ Public 개선 (D_noFamily CV↑ → Public 동점)
- D_minimal이 현재 최적점 (10개 = Age, Fare, FamilySize + Pclass, Sex, Title, Has_Cabin, Pclass_Sex + OOF×2)

## v14~v17 Public Score 결과 (2026-03-23)

| 파일 | CV | Public | 결과 |
|---|---|---|---|
| v14 D_noFamily (9개) | 84.65% | 0.78708 | ❌ FamilySize 제거 = 악화 |
| v16 LogFare only | 84.70% | 0.78708 | ❌ 동점 |
| v17 LogFare+noCabin (8개) | 84.77% | 0.78229 | ❌ 최악 |
| v17 pseudo t=0.85 | (누수) | 0.78468 | ❌ |
| v17 pseudo t=0.80 | (누수) | 0.78708 | ❌ 동점 |

**교훈**: Feature pruning 이후 D_minimal이 sweet spot. 더 줄이면 악화. CV는 신뢰 불가.

제출 예정 완료

## v14 실험 결과 (2026-03-23)

### Phase 1: D_minimal 개별 피처 ablation (5-seed → 10-seed)

| 설정 | 5-seed CV | diff | 10-seed CV | 비고 |
|---|---|---|---|---|
| D_base (10개) | 84.69% | 기준 | - | D_minimal 재확인 |
| **D_noFamily (9개)** | **84.98%** | **+0.29%p ★** | **84.65% ±0.55%** | **FamilySize 제거** |
| D_noCabin (9개) | 84.78% | +0.09%p △ | - | 미선택 (노이즈) |
| D_noTitle (9개) | 84.65% | -0.04%p | - | 실패 |
| D_noAge (9개) | 83.37% | -1.01%p ❌ | - | 치명적 하락 |
| D_noFare (9개) | 83.68% | -0.70%p ❌ | - | 치명적 하락 |

**D_noFamily 10-seed**: 84.65% ±0.55%, diff=+0.0027 vs D_minimal ★ → submission_v14_d_nofamily_10s.csv 저장

### Phase 2: D_minimal k값 재탐색 (5-seed)

모든 k 조합 ±0.001 이내 → **k값 변경 효과 없음** (D_minimal 기준에서도 v10 k_ss=3, k_tick=5 유지)

**교훈**:
- FamilySize는 D_minimal 수준에서 노이즈 — OOF(Ticket/SexSurname)이 가족 정보를 이미 포착
- Age, Fare는 제거하면 치명적 (-0.7~1.0%p) — 핵심 수치 피처
- Title, Has_Cabin 제거는 미미한 영향 (±0.04%p) — 경계선

D_noFamily 피처:
```python
NUM = ["Age", "Fare"]
CAT = ["Pclass", "Sex", "Title", "Has_Cabin", "Pclass_Sex"]
OOF = [SexSurname_k3, Ticket_k5]  # 총 9개
```

## v14b 실험 결과 (2026-03-23)

### D_noFamily 추가 ablation — 모두 실패
| 설정 | 5-seed CV | diff | 비고 |
|---|---|---|---|
| NF_base (기준) | 84.98% | - | D_noFamily |
| NF_noCabin | 85.05% | +0.07%p △ | 노이즈 수준 |
| NF_coreOnly (Age,Fare+Pclass,Sex,Title) | 85.03% | +0.05%p △ | 노이즈 |
| NF_noTitle | 84.72% | -0.27%p | 실패 |
| NF_noTitleCabin | 84.63% | -0.36%p | 실패 |

**교훈**: D_noFamily가 현재 local optimum. Title, Has_Cabin 제거는 노이즈 수준이지만 일관된 개선 없음.

## v15 실험 결과 (2026-03-23)

### Pclass×Sex OOF + SVM 다양성 실험 — 모두 실패
| 설정 | 5-seed CV | diff | 비고 |
|---|---|---|---|
| NF_base (기준) | 84.98% | - | |
| A_PcSex_OOF_replace | 84.67% | -0.31%p ❌ | Pclass_Sex→OOF 교체 |
| B_PcSex_OOF_add | 84.51% | -0.47%p ❌ | Pclass_Sex 유지+OOF 추가 |
| C_PcSex_SVM | 84.47% | -0.52%p ❌ | OOF replace+SVM |
| D_NF_SVM | 84.78% | -0.20%p ❌ | SVM 추가만 |

**교훈**:
- Pclass×Sex OOF는 이미 Sex/Pclass/Pclass_Sex categorical 피처와 중복 → 노이즈
- D_noFamily 피처셋에서 SVM 추가는 효과 없음

## v16 실험 결과 (2026-03-23)

### 피처 변환 + 메타러너 변경 (5-seed)
| 설정 | 5-seed CV | diff | 10-seed CV | 비고 |
|---|---|---|---|---|
| NF_base (기준) | 84.98% | - | - | |
| **A_LogFare** | **85.10%** | **+0.12%p ★** | **84.70% ±0.60%** | Fare→LogFare |
| B_AgePclass | 84.65% | -0.00%p | - | Age×Pclass 추가 |
| C_meta_LR_C001 | 84.20% | -0.79%p ❌ | - | 과도 정규화 |
| D_meta_LR_C01 | 84.94% | -0.04%p | - | 약간 하락 |
| E_meta_RF | 84.83% | -0.16%p | - | 노이즈 |
| F_meta_GB | 84.06% | -0.92%p ❌ | - | 실패 |

LogFare 10-seed: 84.70% ±0.60%, diff=+0.0005 △ — 노이즈 경계

## v17 실험 결과 (2026-03-23)

### A: LogFare + noCabin (8개 피처)
- 10-seed CV: **84.77% ±0.53%**, diff=+0.0012 ★
- 저장: submission_v17_a_logfare_nocabin_10s.csv

### B: Pseudo-labeling (D_noFamily 기준)
⚠️ **CV 누수 경고**: pseudo-label이 전체 훈련셋으로 생성, 해당 테스트 샘플이 CV val fold에 포함 → CV 수치 신뢰 불가. Public score로만 평가.

| threshold | pseudo 개수 | 5-seed CV (누수) | 생존 예측 |
|---|---|---|---|
| 0.90 | 0개 | - | - |
| 0.85 | 182개 (53생존+129사망) | 86.99% ★★ (누수) | 140명 |
| 0.80 | 279개 (83생존+196사망) | 88.60% ★★ (누수) | 137명 |

저장: submission_v17_pseudo85_10s.csv, submission_v17_pseudo80_10s.csv
→ Public score 결과 요망

**교훈**: Pseudo-labeling CV는 반드시 원본 레이블 샘플에서만 평가해야 함.

## 제출 대기 파일 (2026-03-23 기준)

| 파일 | 10-seed CV | 비고 | 우선순위 |
|---|---|---|---|
| submission_v14_d_nofamily_10s.csv | 84.65% ±0.55% | FamilySize 제거 | 1 |
| submission_v17_a_logfare_nocabin_10s.csv | 84.77% ±0.53% | LogFare+noCabin | 2 |
| submission_v17_pseudo85_10s.csv | - (CV누수) | pseudo-label t=0.85 | 3 |
| submission_v17_pseudo80_10s.csv | - (CV누수) | pseudo-label t=0.80 | 4 |
| submission_v16_a_logfare_10s.csv | 84.70% ±0.60% | LogFare only | 5 |
| submission_v13_d_minimal_10s.csv | 84.38% ±0.58% | D_minimal | 6 |

## v18 threshold 최적화 결과 (2026-03-23)

D_minimal 10-seed 기준 threshold 탐색:
| threshold | 생존 예측 | Public |
|---|---|---|
| **0.50** | **137명** | **0.79186 ★** |
| 0.45 | 152명 | 0.77511 ❌ |
| 0.39 | 160명 | 0.76076 ❌ |

**교훈**: threshold 0.50 이미 최적. 생존 예측 수 늘리면 급격히 악화.
(훈련셋 기준 기대값 160명은 테스트셋에 적용 안 됨)

## 2026-03-23 오늘의 전체 교훈

| 발견 | 내용 |
|---|---|
| 피처 pruning | 19개→10개 = +0.00478 Public ★★★ |
| FamilySize 제거 | 0.78708 (D_minimal과 동점) → 핵심 피처 |
| Feature pruning sweet spot | D_minimal(10개): Age,Fare,FamilySize+Pclass,Sex,Title,Has_Cabin,Pclass_Sex+OOF×2 |
| CV vs Public | D_noFamily(CV↑) = D_minimal(Public↑) → CV는 방향 참고만 |
| Threshold | 0.50이 최적. 생존 수 늘리면 급하락 |
| Pseudo-labeling | CV 누수로 무효, Public에서도 개선 없음 |

## v19 실험 결과 (2026-03-24)

### 개별 피처 추가 (5-seed quick scan)
| 설정 | 5-seed CV | diff | 비고 |
|---|---|---|---|
| base (기준) | 84.XX% | - | D_minimal |
| +Cabin_Deck | best △ | +0.0007 △ | 경계선 |
| +SibSp, +Parch 등 | - | <+0.001 | 유의미 개선 없음 |

10-seed 검증 미진행 (5-seed 유망 후보 없음)

## v19b 실험 결과 (2026-03-24) — 10-seed 전체 검증

| 설정 | 10-seed CV | std | 생존 | Public |
|---|---|---|---|---|
| Dmin_base | 84.38% | ±0.58% | 137 | 0.79186 |
| **Dmin+CabinDeck** | **84.57%** | **±0.26%** | 138 | **0.79425 ★** |
| Dmin+AgeGroup | 84.32% | ±0.53% | 136 | 0.78947 ❌ |
| **Dmin+FamGrp** | **84.38%** | **±0.62%** | 136 | **0.79425 ★** |
| Dmin+SVM | 84.47% | ±0.39% | 138 | 0.78947 ❌ |
| v10_full | 83.77% | ±0.37% | 137 | 0.79186 |
| blend60(Dmin)+40(v10) | - | - | 136 | 0.79425 ★ |
| blend70(Dmin)+30(v10) | - | - | 136 | 0.79425 ★ |

**핵심 발견**: Cabin_Deck 추가 및 FamilySizeGroup 추가 둘 다 0.79425로 동점 신기록.

## v20 실험 결과 (2026-03-24) ⭐ 신기록!

### DC_F (Dmin + CabinDeck + FamilySizeGroup) = **0.79665**

| 설정 | 10-seed CV | std | 생존 | Public |
|---|---|---|---|---|
| Dmin_base | 84.38% | ±0.58% | 137 | 0.79186 |
| DC (Dmin+CabinDeck) | 84.57% | ±0.26% | 138 | 0.79425 |
| DF (Dmin+FamGrp) | 84.38% | ±0.62% | 136 | 0.79425 |
| **DC_F (Dmin+CabinDeck+FamGrp)** | **84.48%** | **±0.40%** | 137 | **0.79665 ★★★** |
| DC_SVM (DC+SVM) | 84.55% | ±0.34% | 140 | 0.79425 |

**DC_F 피처셋** (12개):
```python
NUM = ["Age", "Fare", "FamilySize"]
CAT = ["Pclass", "Sex", "Title", "Has_Cabin", "Pclass_Sex", "Cabin_Deck", "FamilySizeGroup"]
OOF = [SexSurname_k3, Ticket_k5]
```

**핵심 교훈**: 두 피처를 각각 추가하면 +0.00239씩이지만 **같이 추가하면 시너지 효과**로 +0.00480. Cabin_Deck과 FamilySizeGroup은 상호보완적.

DC×DF 블렌드: 모두 0.79425 (137 surv) — DC_F보다 낮음.

## v21 실험 결과 (2026-03-25)

### DC_F 기반 추가 피처 확장 — 모두 baseline 이하

| 설정 | 10-seed CV | std | 생존 | diff |
|---|---|---|---|---|
| DCF_base (기준) | 84.48% | ±0.40% | 137 | 기준 |
| DCF_SVM | 84.51% | ±0.38% | 139 | +0.0003 △ |
| DCF+IsAlone | 84.50% | ±0.36% | 137 | +0.0002 △ |
| DCF+AgeGrp | 84.49% | ±0.50% | 137 | +0.0001 △ |
| DCF+SibSp | 84.27% | ±0.50% | 138 | -0.0021 ❌ |
| DCF+Embarked | 84.15% | ±0.49% | 136 | -0.0033 ❌ |

**교훈**: DC_F에서 추가 피처는 모두 유의미한 개선 없음 (0.001 미만 차이). SibSp/Embarked는 오히려 악화. DC_F(12개)가 현재 sweet spot.

### DCF × DCF_SVM 블렌드
| 비율 | 생존 | 비고 |
|---|---|---|
| DCF60 + DCF_SVM40 | 137 | |
| DCF70 + DCF_SVM30 | 137 | |
| DCF80 + DCF_SVM20 | 137 | |

Public 점수 미확인 (제출 대기 중)

## 다음 실험 후보 (2026-03-25 이후)

### 1순위 — DC_F 기반 새 방향
- DC_F + Parch 조합 (SibSp는 실패했지만 Parch는 미시도)
- DC_F + LogFare (FamilySize와 LogFare 조합)
- DC_F + TicketFreq
- DC_F 기반 2-level stacking (meta를 RF/GB로)

### 2순위 — 다른 피처셋 탐색
- CatBoost 추가 (베이스 모델 다양성)
- 5-fold → 10-fold CV 구조 변경
- Has_Cabin 대신 Cabin_Deck만 (Cabin_Deck이 Has_Cabin 정보 포함)

### 3순위 — 블렌딩 전략
- DC_F × DC (138 surv) 블렌드
- DC_F × DF (136 surv) 블렌드
- 다른 생존자 수 예측값과 블렌딩

---

## 제출 전 체크리스트

- [ ] 10-seed 반복 CV로 안정성 확인
- [ ] Stacking meta-CV 기준으로 판단 (solo CV 과대평가)
- [ ] 현재 최고(83.88%) 대비 명확한 개선 (0.5%p 이상)
- [ ] 생존 예측 수 확인 (기대값 160명 ± 15명)
- [ ] 이전 실패 패턴과 다른지 확인

---

## 파라미터 현황 (v9 기준)

```python
# 피처 (19개)
NUM_FEATURES = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "TicketFreq", "LogFare"]
CAT_FEATURES = ["Pclass", "Sex", "Embarked", "Title", "IsAlone", "Has_Cabin",
                "Cabin_Deck", "FamilySizeGroup", "AgeGroup", "Pclass_Sex"]
OOF_FEATURES = ["SexSurname_OOF (k=3)", "Ticket_OOF (k=5)"]

# 모델
XGB_PARAMS = dict(n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)
LGBM_PARAMS = dict(n_estimators=500, num_leaves=31, learning_rate=0.05,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, random_state=42)
META = LogisticRegression(C=1.0, max_iter=500)
```
