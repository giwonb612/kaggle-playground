# Titanic Kaggle Playground

이 폴더는 Kaggle Titanic 대회의 실험 기록과 최종 제출 파이프라인을 정리한 프로젝트다.

## Final Result

- 최종 제출 기준 모델: `v20`의 `DC_F`
- 최종 Public Score: **0.79665**
- 10-seed 평균 CV: **0.8448** (`84.48%`)
- 제출 파일: `data/submissions/submission_v20_dc_f_10s.csv`
- 기준 문서: `EXPERIMENTS.md`

`DC_F`는 `D_minimal` 피처셋에 `Cabin_Deck`과 `FamilySizeGroup`을 추가한 최종 조합이다.

## Final Model

### 1) Feature Set

최종 피처 수는 총 12개다.

```python
NUM = ["Age", "Fare", "FamilySize"]
CAT = ["Pclass", "Sex", "Title", "Has_Cabin", "Pclass_Sex", "Cabin_Deck", "FamilySizeGroup"]
OOF = ["SexSurname_k3", "Ticket_k5"]
```

핵심 포인트:

- `Age`: `Pclass x Title` 그룹 median으로 결측 보정
- `Title`: 이름에서 추출
- `Has_Cabin`, `Cabin_Deck`: 객실 보유 여부와 deck 정보 분리 사용
- `Pclass_Sex`: 객실 등급과 성별의 상호작용
- `FamilySizeGroup`: `Alone / Small / Large`
- `SexSurname_k3`: `Sex + Surname` 기준 Bayesian-smoothed OOF target encoding
- `Ticket_k5`: `Ticket` 기준 Bayesian-smoothed OOF target encoding

### 2) Algorithm

최종 알고리즘은 2단계 stacking ensemble이다.

Base models:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM low-learning-rate variant

Meta learner:

- Logistic Regression (`C=0.05`)

구조:

1. 5-fold stratified CV로 각 base model의 OOF probability 생성
2. OOF probability들을 meta feature로 사용
3. Logistic Regression meta learner로 최종 예측
4. 서로 다른 10개 seed 결과를 평균내어 submission 생성

즉, 최종 제출은 단일 모델이 아니라 `5개 base model + LR meta learner + 10-seed averaging` 조합이다.

## Score Summary

| Version | 핵심 아이디어 | CV | Public |
|---|---|---:|---:|
| v10 | stacking 구조 개선, low-LR LGBM 도입 | 84.12% | 0.78708 |
| v13 D_minimal | 피처 축소(19개 → 10개) | 84.38% | 0.79186 |
| v19b Dmin+CabinDeck | `Cabin_Deck` 추가 | 84.57% | 0.79425 |
| v19b Dmin+FamGrp | `FamilySizeGroup` 추가 | 84.38% | 0.79425 |
| v20 DC_F | `Cabin_Deck + FamilySizeGroup` 동시 추가 | 84.48% | **0.79665** |

해석:

- `Cabin_Deck` 단독 추가와 `FamilySizeGroup` 단독 추가도 각각 성능 향상이 있었다.
- 둘을 동시에 넣은 `DC_F`가 가장 높은 Public Score를 기록했다.
- 이후 `v21`, `v22`에서 추가 피처와 변형을 더 실험했지만 `0.79665`를 넘지 못했다.

## Why This Model Won

이 프로젝트에서 가장 중요했던 결론은 아래 4가지다.

1. 피처를 많이 넣는다고 좋아지지 않았다.
2. 작은 데이터셋에서는 강한 튜닝보다 피처 pruning이 더 효과적이었다.
3. `Sex + Surname`, `Ticket` 기반 OOF 인코딩이 가장 강력한 추가 신호였다.
4. 단일 모델보다 stacking ensemble이 일관되게 우수했다.

특히 `D_minimal`로 불필요한 피처를 제거한 뒤,
`Cabin_Deck`과 `FamilySizeGroup`을 더했을 때 일반화 성능이 가장 좋았다.

## Experiment Timeline

- `v1`~`v9`: baseline 구축, OOF survival encoding 도입, stacking 구조 안정화
- `v10`: low-LR LGBM base model 도입, Public `0.78708`
- `v13`: minimal feature set 확정, Public `0.79186`
- `v19b`: `Cabin_Deck`, `FamilySizeGroup` 각각 유효함 확인, Public `0.79425`
- `v20`: 두 피처를 동시에 사용한 `DC_F`로 최종 최고점 `0.79665`
- `v21`~`v22`: 추가 확장 실험 진행, 최고 기록 갱신 실패

## Important Files

- `run_v20.py`: 최종 최고점 `DC_F` 실험 및 제출 생성
- `run_v21.py`: `DC_F` 기반 추가 실험
- `run_v22.py`: `DC_F` 주변 sweet spot 탐색
- `EXPERIMENTS.md`: 전체 실험 로그와 public score 기록
- `src/features.py`: feature engineering
- `src/config.py`: 공통 설정

## Reproduce

의존성 설치:

```bash
pip install -r requirements.txt
```

최종 모델 실행:

```bash
python run_v20.py
```

생성 결과:

- `data/submissions/submission_v20_dc_f_10s.csv`

## Notes

- `v17` pseudo-labeling 실험은 CV 누수 문제가 있어 최종 모델에서 제외했다.
- threshold tuning도 시도했지만 `0.50`이 최적이었다.
- 최종 선택 기준은 local CV만이 아니라 실제 Kaggle Public Score였다.
