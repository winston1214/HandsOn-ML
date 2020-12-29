# 한눈에 보는 머신러닝

## 머신러닝이란?

- 머신러닝은 데이터로부터 학습하도록 컴퓨터를 프로그래밍하는 과학
- 명시적인 프로그래밍 없이 컴퓨터가 학습하는 능력을 갖추게 하는 연구분야

**예시** : 스팸 필터
  - **Training Set** : 시스템이 학습하는데 사용하는 ***샘플***
  - **Tranining Data** : 이 메일은 스팸이다라는 ***경험***
  - **Accuracy** : 성능 측정(정확도)
  
## 왜 머신러닝을 사용하는가?
<img src='https://tensorflowkorea.files.wordpress.com/2018/05/e18489e185b3e1848fe185b3e18485e185b5e186abe18489e185a3e186ba-2018-05-23-e1848be185a9e18492e185ae-11-41-57.png?w=768'></img>

오차 분석 등을 통해 새로운 오차들을 학습시켜서 더 좋은 예측 성능을 낼 수 있음

**데이터 마이닝(Data Mining)** : 머신러닝 기술을 적용해서 대용량의 데이터를 분석하면 겉으로 보이지 않던 패턴을 발견할 수 있음

<img src='https://tensorflowkorea.files.wordpress.com/2018/05/e18489e185b3e1848fe185b3e18485e185b5e186abe18489e185a3e186ba-2018-05-23-e1848be185a9e18492e185ae-11-47-00.png?w=768'></img>

- 머신러닝을 사용했을 때 더욱 효율적인 분야
```
1. 기존 솔루션으로는 많은 수동 조정과 규칙이 필요한 문제
2. 전통적인 방식으로는 전혀 해결 방법이 없는 복잡한 문제
3. 유동적인 환경
4. 복잡한 문제와 대량의 데이터에서 인사이트 얻기
```

## 머신러닝 시스템의 종류

- **지도학습**

**지도학습(Supervised Learning)** 이란 알고리즘에 주입하는 훈련 데이터에 **레이블(label)** 이라는 답이 포함되어 학습

<img src='https://nohjiho.github.io/images/ml/tensorflow/%EC%A7%80%EB%8F%84%ED%95%99%EC%8A%B5_%EB%A0%88%EC%9D%B4%EB%B8%94%EB%90%9C%ED%9B%88%EB%A0%A8%EC%84%B8%ED%8A%B81.png'></img>

  - **분류(Classification)** : 전형적인 지도학습의 예( ex. 스팸이다 아니다 분류 예측)
  - **회귀(Regression)** : **예측 변수**를 **특성**을 이용하여 **타겟 수치**를 예측 ( ex. 주행거리, 연식, 브랜드로 중고차의 가격 예측)
  
  ### 지도학습 알고리즘 예시:
    - KNN(K-Nearest Neighbors)
    - Linear Regression
    - Logistic Regression
    - Support Vector Machine(SVM)
    - Decision Tree & RandomForest
    - Neural Networks

- **비지도학습**

  **비지도학습**이란 훈련 데이터에 레이블이 없음(지도학습과 반대)
  <img src='https://nohjiho.github.io/images/ml/tensorflow/%EB%B9%84%EC%A7%80%EB%8F%84%ED%95%99%EC%8A%B5.png'></img>

  ### 비지도학습 알고리즘 예시:
    - Clustering
      - KMeans Clustering
      - Hierarchical Cluster Anaylsis(HCA)
      - Expectation Maximization
    - Visualization & Dimensionality Reduction
      - PCA
      - kernelPCA
      - LLE(Locally-Linear Embedding)
      - t-SNE
    - Association rule learning
      - Apriori
      - Eclat
      
  - **군집 알고리즘(Clustering)** : 각 사용자의 특성을 고려하여 군집화 시킴
  - **시각화** : 시각화를 통해서 데이터의 조직을 파악(EDA)
  - **차원 축소** : 많은 정보를 잃지 않으면서 데이터를 간소화 하는 작업(ex. 상관관계가 있는 여러 특성을 하나로 합침 - Feature Extraction)
  - **이상치 탐지** : 정상 샘플로 훈련하여 새로운 샘플이 정상인지 아닌지를 판단
  - **연관 규칙 학습** : 장바구니 분석 등 대량의 데이터에서 특성 간의 유의미한 관계를 찾는 것

- **준지도학습**

  **준지도학습**이란 레이블이 일부만 있는 데이터. 보통은 레이블이 없는 데이터가 많고 레이블이 있는 데이터는 적음(ex. 구글 포토 호스팅 서비스)
  ### 비지도학습 알고리즘 예시: (비지도 + 지도)
    - DBN(Deep Belif Network) -> RBM(Restricted Boltzmann Machine)이란 비지도 학습에 기초, RBM으로 훈련 후 지도학습으로 세밀 조정
<img src='https://nohjiho.github.io/images/ml/tensorflow/%EC%A4%80%EC%A7%80%EB%8F%84%ED%95%99%EC%8A%B5.png'></img>

- **강화학습**

  **강화학습** 이란 **에이전트**라는 학습하는 시스템에서 환경을 관찰해서 행동을 하고 그 결과로 **보상(Reward)** 또는 **벌점(Penalty)** 을 받아 최적의 **정책(Policy)** 을 수립

<img src='https://tensorflowkorea.files.wordpress.com/2018/05/e18489e185b3e1848fe185b3e18485e185b5e186abe18489e185a3e186ba-2018-05-24-e1848be185a9e1848ce185a5e186ab-12-21-44.png?w=768'></img>
  
  

  
