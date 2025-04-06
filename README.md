# 2-sophie-kim-personal-project
# Rice Image Classification with Transfer Learning

## 프로젝트 개요
이 프로젝트에서는 쌀 품종 이미지 분류를 위해 VGG16과 EfficientNet-B0 두 가지 사전 훈련된 모델을 비교하고, 데이터셋 크기에 따른 성능 변화를 분석했습니다.

## 데이터셋
- **Rice Image Dataset** (Kaggle)
- 여러 쌀 품종의 고품질 이미지로 구성

## 주요 기능
- 사전 훈련된 VGG16 및 EfficientNet-B0 모델의 전이학습 구현
- 다양한 데이터셋 크기(1%, 5%, 50%, 100%)에서의 성능 비교
- 모델 성능 평가 및 시각화
- SQLite 데이터베이스를 통한 실험 결과 저장 및 조회

## 설치 방법
1. 필요한 라이브러리 설치:
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn pillow
```

2. Kaggle API 설정 (데이터셋 다운로드용):
```bash
pip install kaggle
```
이후 kaggle.json 파일을 ~/.kaggle/ 경로에 복사하고 적절한 권한 설정 필요

## 사용 방법
1. 데이터셋 다운로드 및 준비:
```python
!kaggle datasets download -d muratkokludataset/rice-image-dataset
!unzip rice-image-dataset.zip -d rice-image
```

2. 모델 학습 실행:
```python
# VGG16 또는 EfficientNet 모델 학습 코드 실행
```

3. 결과 조회:
```python
view_experiments_db()  # 데이터베이스에 저장된 실험 결과 조회
```

## 주요 파일 설명
- `rice_classification_vgg16.py`: VGG16 모델을 사용한 쌀 이미지 분류 구현
- `rice_classification_efficientnet.py`: EfficientNet-B0 모델을 사용한 쌀 이미지 분류 구현
- `model_performance.db`: 실험 결과가 저장되는 SQLite 데이터베이스

## 모델 구조
1. **VGG16**:
   - 마지막 4개 컨볼루션 레이어만 훈련 가능하게 설정
   - 분류기 부분 수정 (512*7*7 → 4096 → 512 → 클래스 수)
   - 학습률: 0.0001

2. **EfficientNet-B0**:
   - 마지막 2개 블록만 훈련 가능하게 설정
   - 분류기 부분 수정 (1280 → 클래스 수)
   - 학습률: 0.00005

## 학습 설정
- 배치 크기: 256
- 최대 에폭 수: 50
- 조기 종료 : 5
- 손실 함수: CrossEntropyLoss
- 옵티마이저: Adam
- 학습률 스케줄러: ReduceLROnPlateau

## 결과 분석
데이터베이스에 저장된 결과를 통해 다음과 같은 분석이 가능합니다:
- 모델별 테스트 정확도 비교
- 데이터셋 크기에 따른 성능 변화 추이
- 클래스별 정확도 및 혼동 행렬 분석
- 학습 손실 및 정확도 그래프 시각화

