# level2_objectdetection-cv-12

## 🔍 프로젝트 개요
    
본 프로젝트는 10종류의 쓰레기가 찍힌 데이터셋에 대하여 object detection을 수행하는 우수한 성능의 모델을 개발함으로써,  쓰레기 분리수거를 돕거나 어린아이들의 분리 수거 교육에 도움이 되는 것을 목표로 진행되었다.
    
- **Input :** 쓰레기 객체가 담긴 이미지가 모델의 인풋으로 사용된다. 또한 bbox 정보(좌표, 카테고리)는 model 학습 시 사용이 된다. bbox annotation은 COCO format으로 제공된다.
- **Output :** 모델은 bbox 좌표, 카테고리, score 값을 반환한다.

## 👨‍🌾 프로젝트 팀 구성 및 역할

- level2 object detection team 12 - **GANDDDDI**
- **도환** : stratifiedGroupKfold, augmentation 실험, 모델 예측 결과 분석, 여러 모델 실험
- **아라** : bbox 시각화 코드 작성, ensemble 코드 작성 및 실험, 모델 서치 및 학습
- **무열** : 백본 모델 서치 및 학습, 코드 실행 시 발생하는 에러들 해결
- **성운** : RedisDB를 통한 GPU 관리, Large-scale model 훈련, fp16 훈련
- **현민** : EDA를 통한 다양한 아이디어 고안 및 구현, 여러 모델 서치 및 학습


## 🔥 수행 내용 🔥

1. RedisDB를 통한 GPU 관리
    - work queue 기능을 사용하여 GPU utilization 향상
    - 7.194 days * 5 people 사용량 달성
2. EDA를 통해 학습데이터 분석
    - 한 이미지에 30/40개 이상의 박스가 존재하는 경우 제거
3. mmdetection 구조 분석 
    - 실험에 주로 활용할 configs를 중심으로 분석
4. Stratified K-fold를 통해 validation fold 탐색
    - val mAP가 가장 높고, leader board mAP와 큰 차이가 나지 않는 validation fold를 선정
5. 1 stage, 2 stage 모델 비교 및 최종 모델 선정 과정
    - yolov8, yolox, Faster R-CNN ResNet 50,  Faster R-CNN ResNeXt 101, DetectoRS Cascade RCNN ResNet50, Cascade ResNeXt 101, Cascade swin -T, Cascade swin-B, Cascade dynamic head ConvNeXt, ATSS Swin-L
6. mixed precision training 적용
    - fp16 방식으로 메모리, 시간 2배 줄이기
7. Augmentation 실험 진행
    - Flip, Blur, Mosaic 등 최적의 augmentation 조합 탐색
8. 모델 예측 결과 분석
    - 모델이 잘못 예측한 결과를 분석해서 얻은 정보를 활용하여 모델 성능 향상 시도
9. 성능 향상을 위한 접근 방법 
    - ensemble(WBS, NMS 등) 수행
    - pseudo labeling
    - submission(.csv) 파일을 기반으로 ensemble을 진행하는 ensemble.py 파일 작성 및 활용

## 🧱 협업 문화

- 협업 tool : Github, WandB, Notion, kakaoTalk, zoom, slack, Gathertown
    - Github : mmdetection 모델 실험 코드 관리, redisdb publisher, consumer 코드 공유
    - WanDB : 모델 실험 결과 공유, Val mAP, Train Loss 비교
    - Notion :  진행사항 기록, 모델 훈련 중 발생한 error 기록

## 🧪 Experiments

|  | Model | Backbone | val mAP  | Setting |
| --- | --- | --- | --- | --- |
| 1 Stage | Yolov8 | YOLOv8 | 0.3093 | Epoch 300, Batchsize 16, Learning rate 0.01 |
|  | Yolov3 | DarkNet-53 | 0.104 | Epoch 5 , Batchsize 8, Learning rate 0.001 |
|  | Yolovx | Modified CSPv5 | 0.125 | Epoch 29 , Batchsize 8, Learning rate 0.001 |
|  | EffcientDet_b3 |  | 0.403 | Epoch 50 , Batchsize 8, Learning rate 0.005 |
| 2 Stage | Cascade R-CNN | Swin Base | 0.624 | Epoch 32 , Batchsize 4, Learning rate 0.0001 |
|  | Cascade R-CNN | Swin Tiny | 0.495 | Epoch12, Batchsize 8, Learning rate 0.0001 |
|  | Cascade R-CNN (dynamic head) | ConvNeXt | 0.651 | Epoch 17 , Batchsize 2, Learning rate 0.0001 |
|  | Faster R-CNN | ResNet50 | 0.42 | Epoch12, Batchsize 4, Learning rate 0.02 |
|  | Faster R-CNN | ResNeXt101 | 0.462 | Epoch12, Batchsize 16, Learning rate 0.02 |
|  | DetectoRS | ResNet50 | 0.472 | Epoch12, Batchsize 16, Learning rate 0.02 |
|  |  |  |  |  |

## 최종 모델

augmentation 기반으로 Cascade R-CNN 모델에서 백본을 Swin Base, ConvNeXt로 구성하여, 최종적으로 WBF로 앙상블을 진행하였다.

- Augmentation
    - Oneof (RandomFlip, HorizontalFlip, VerticalFlip)
    - Normalize
    - RandomRotate90
    - diagonal
    - Oneof (RandomBrightnessContrast, CLAHE)
    - HueSaturationValue
    - Oneof (GaussianBlur, MedianBlur, MotionBlur)
    - Mosaic
    - Multi-Scale
- Optimzer, Scheduler
    - AdamW(Swin Base, ConvNeXt)
    - StepLR(Swin Base)
    - CosineAnnealing(ConvNeXt)
- Model
    | Model | Backbone | K-Fold Val | Eval mAP(Public) |
    | --- | --- | --- | --- |
    | Cascade R-CNN | Swin Base | fold0 | 0.6526 |
    |  |  | fold1 | 0.6474 |
    |  |  | fold2 | 0.655 |
    |  |  | fold3 | 0.6596 |
    |  |  | fold4 | 0.6559 |
    | Cascade R-CNN (dynamic head) | ConvNeXt | fold0 | - |
    |  |  | fold1 | - |
    |  |  | fold3 | 0.66 |
    - Cascade R-CNN + Swin Base
        - Epoch 32 , Batchsize 4, Learning rate 0.0001
    - Cascade R-CNN(dynamic head) + ConvNeXt
        - Batchsize 2, Learning rate 0.0001
        - 시간 관계상 fold 마다 Epoch 11, 14, 17를 사용하였다.
    - 아래 그림은 최종으로 사용한 8개의 모델의 학습 과정 중 validation set의 mAP 그래프이다. x 축이 Step, y축이 val mAP이다.

- WBF Ensemble (IoU Threshold 0.6)
      위 8개의 Ensemble 결과 : **0.7047**


## 🏆Result
- 총 19 팀 참여
- Public : 3등 -> Private : 3등
- Public : 0.7047 -> Private : 0.6881

