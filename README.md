# 2024_KAMP 경진대회

## Concept Drift 탐지와 Active Learning을 통한 실시간 불량 탐지

- 제4회 K-인공지능 제조데이터 분석 경진대회(다이캐스팅 주조 공정 불량 탐지)에서 수행한 코드와 실험 결과를 정리한 저장소
- 주조 공정에서 수집되는 공정 데이터를 기반으로, concept drift 탐지 + 온라인 학습 + Active Learning을 결합한 실시간 불량 탐지 프레임워크를 구현

### 1. 프로젝트 개요

- 다이캐스팅 주조 공정에서는 불량이 주로 후공정에서 발견되어, 원인 파악 및 조치가 늦어지고 대량 불량으로 이어질 수 있음

- 이를 해결하기 위해 주조 단계에서 수집한 센서 데이터로 실시간 불량 예측 모델을 구축

- 데이터 분포가 시간에 따라 변하는 concept drift 환경을 전제로 

    - 비지도 drift 탐지기 D3 (Discriminative Drift Detector)

    - 온라인 업데이트가 가능한 분류기들(Adaptive Random Forest, MLP, ONN)

    - 라벨링 비용을 줄이기 위한 Active Learning 전략(Random / Entropy / K-means) 하나의 프레임워크로 통합해 실험


### 2. 주요 아이디어
#### Concept Drift 탐지 (D3)

- 버퍼 내 과거 데이터 vs 최신 데이터를 이진 분류 문제로 구성

- Logistic Regression의 AUC가 임계값을 넘으면 분포 변화가 유의미하다고 판단하고 drift로 간주

#### 온라인 분류 모델

- Adaptive Random Forest, MLP, Online Neural Network(ONN)을 사용해
drift 탐지 시 버퍼에 쌓인 최신 데이터를 이용해 순차적으로 파라미터를 업데이트하는 구조를 구현

- Offline vs Online 성능을 시간 구간별 F1-score로 비교해, 드리프트 이후 성능 회복 여부를 분석

##### Active Learning

- 모든 데이터를 라벨링하는 대신, **예측 확률의 불확실성(entropy)에 기반한 샘플 선택**, **피처 공간에서 K-means 기반 대표 샘플 선택** 전략을 사용해 제한된 라벨 예산 하에서 성능을 최대화하는 방법을 실험

- 특히 ONN_online_AL.ipynb에서 `ONN` 모델을 대상으로 **라벨 비율(20/40/60/80%)과 전략(Random / Entropy / K-means)별 성능**을 비교

#### 평가 방법

- 테스트 데이터를 시간 순서로 일정 구간(예: 30개 Chunk)으로 나누고, 구간별 F1-score를 비교

- Offline vs Online 모델, 그리고 Active Learning 전략과 라벨 비율에 따른 성능 변화를 시계열 관점에서 평가

- Friedman / Nemenyi 검정과 critical difference diagram을 이용해 전략 간 성능 차이의 통계적 유의성을 검증

### 3. 저장소 구조

- `EDA_teamproject.ipynb`

    - 주조 공정 데이터 EDA 및 전처리

    - 변수 분포, 상관관계, 불량률 변화 등 기본 분석 수행

- `Vis_ConceptDrift_performance.ipynb`

    - 시간 구간별 분포 변화 및 성능 변화를 시각화

    - Kernel PCA + SVM decision boundary, 구간별 F1-score 라인 플롯 등을 포함

- `Adaptive random forest.ipynb` / `MLP.ipynb` / `ONN.ipynb`

    `- Offline / Online 학습 실험 노트북

    - concept drift 탐지 결과(D3)에 따라 버퍼 데이터를 사용해 모델을 업데이트

    - 시간 구간별 F1-score, Accuracy, Balanced Accuracy 비교

- `ONN_online_AL.ipynb`

    - Online Neural Network(ONN)를 대상으로 Active Learning 실험을 수행하는 노트북

    - 라벨 비율(20%, 40%, 60%, 80%)과 샘플링 전략(Random / Entropy / K-means)별 F1-score 변화를 비교·시각화

- `D3.py`

    - Discriminative Drift Detector(D3) 구현 스크립트

    - 버퍼 내 과거/최신 데이터를 분류하고 AUC 기반으로 drift 여부를 판단

- `pred_results/`, `test_results/`, `model_params/`

    - 모델별 예측 결과, 구간별 성능 지표, 재현 실험을 위한 설정값과 파라미터 저장

    - Friedman/Nemenyi 검정, CD 다이어그램, Active Learning 비교 그래프 등에 필요한 입력 데이터로 사용

- `onn/`

    - Online Neural Network 구현을 위한 커스텀 모듈

    - pip install onn으로 설치되는 기본 패키지 대신, 이 폴더의 파일들로 교체해 실험 재현을 수행

### 4. 실행 환경

- Python: 3.9.X

- 주요 라이브러리
    - numpy(1.26.4), pandas, scikit-learn, matplotlib, seaborn, scikit-posthocs


### 5. 재현 방법

- 데이터 및 환경 준비

    - 경진대회용 주조 공정최적화 데이터셋.csv를 프로젝트 루트 혹은 노트북에서 지정한 경로에 위치시키기

    - 노트북 상단 import를 참고해 필요한 라이브러리를 설치

- EDA 및 드리프트/성능 시각화

    - `EDA_teamproject.ipynb` 실행: 데이터 확인 및 전처리

    - `Vis_ConceptDrift_performance.ipynb` 실행: concept drift 및 성능 변화 시각화

- 모델 학습 및 온라인 업데이트 실험

    - `Adaptive random forest.ipynb`, `MLP.ipynb`, `ONN.ipynb`를 실행

    - Offline vs Online, drift 발생 전·후 성능 차이 및 회복 양상을 확인

- Active Learning 실험

    - `ONN_online_AL.ipynb`를 실행

    - 라벨 비율과 샘플링 전략에 따른 성능 곡선을 비교·분석


### 6. 기대 효과

- 공정 단계에서 불량을 조기에 탐지하여, 후공정에서만 발견되던 대량 불량을 사전에 방지할 수 있음

- 데이터 분포 변화에 자동으로 적응하는 온라인 모델링을 통해 장기 운영 시에도 안정적인 예측 성능을 유지 가능

- Active Learning을 도입해 라벨링 비용을 최소화하면서도 충분한 성능을 확보할 수 있어, 자원이 제한된 제조 환경에서도 적용 가능