---
title: Hand Gesture/Pose Estimation - 2 (Deep Learning)
tags: ['ml', 'vision']
author: "Dongseok Yang"
---

## 1. Intro

- 앞서 소개드린바와 같이 팜 글러브 기반 손 운동학적 정보를 추출하기 위해 기하학적 접근법을 수행하였습니다
- 하지만 기하학적 접근법은 아래 그림과 같이 한계점을 가지고 있습니다.
    - 손가락이 안으로 심하게 굽은 경우 정확한 손가락 끝점을 찾기 못하였고 마찬가지로 손의 추적선을 재대로 찾지 못하였습니다.
    - 또한 손가락의 외각선 추출이 가능하더라도 정확한 손관절값을 유추하기는 쉽지 않았으며, 심한 적외선 왜곡 현상에 잘 대처하지 못하였습니다.

<img src="https://user-images.githubusercontent.com/37643248/164342849-093ed50a-4f22-42e2-9415-79c601b01a89.png">

- 이러한 문제점들을 해결하기 위해 딥러닝 접근법을 도입하였습니다.

## 2. 손 자세 추정을 위한 딥러닝 네트워크 및 경량화 방법

- 견고하고 가벼운 손 자세 추정을 위해, 팜 글러브를 위한 **손 자세 데이터 생성기**, 올바른 학습 방향을 유도하는 **Patrial Augmentation**, 손 자세 추정을 위한 **딥러닝 네트워크 및 경량화 방법**에 대해 순서대로 말씀드리겠습니다.

<img src="https://user-images.githubusercontent.com/37643248/164342893-668c800b-eacb-482e-ac0c-bd5ceb705c41.png">

### 2.1 손 자세 데이터 생성기

- 먼저 손 자세 데이터 생성기 입니다. 딥러닝에서 가장 중요한 것은 고른 분포를 가진 많은 데이터 세트 입니다. 따라서 공용 손 데이터 세트(Public Database)도 많이 존재합니다.
- 하지만 모든 데이터 세트의 유형은 거치형 카메라 기반으로 수집된 데이터 베이스 입니다. 즉, 팜-글러브에 적용하기에 적합하지 않습니다.
- 공용 손 데이터 세트 중에서 실제 팜 글러브에 사용 가능한 총 데이터의 양은 20퍼센트도 채 되지 않았습니다.

<img src="https://user-images.githubusercontent.com/37643248/164342906-c85283d9-ab3c-4411-9e4a-e8a0b188a266.png">

- 이러한 문제점을 해결하기 위해 저는 손 데이터 생성기를 개발하였습니다.
- 손 데이터 생성기는 옵션 형식으로 손가락 관절 및 손목의 움직임의 범위를 조절할 수 있으며, 카메라의 위치 및 조도 또한 설정이 가능합니다.
- 결과값은 2d, 3d, 및 오일러 앵글과 주석에 해당하는 이미지가 출력됩니다. 또한 생성기의 중요 특징중 하나는 물리엔진이 내장되있어 손가락이 물리적으로 가질 수 없는 모델이 생성되면 회피하거나 심한경우 데이터 세트에서 자동으로 제외됩니다.

<img src="https://user-images.githubusercontent.com/37643248/164342947-c8f3388f-325c-408b-9af7-9a816ba32c74.png">

### 2.2 부분 변조 기법 (Partial Augmentation)

- 부분 변조 기법은 아래 그림과 같이 파란점이 그라운드 쓰루라고 할때 예측된 점이 빨간색으로서 일정한 값 이상 에러를 가질 경우 해당 부분에 대해서만 변조을 수행합니다.
- 이러한 방법은 손 자세 추정에 특히 많은 이점을 가집니다.
- 왜냐하면 손가락은 대부분 비슷하게 생겼으며, 하나의 점의 오류를 보완하기 위해서 다른 점을 예측하는 딥러닝 노드가 같이 수정되면 정확한 예측을 수행했던 점이 다시 일부 재학습되어 에러를 가지게 되는 보틀랙(Bottleneck) 현상이 생기게 되기 때문입니다.

<img src="https://user-images.githubusercontent.com/37643248/164343018-aa70b731-a8e5-4af4-85ec-fa1837700a44.png">

- 처음부터 Partial Augmentation을 수행하지 않고 학습이 80%이상 진행된 뒤에 적용됩니다.
- 학습 과정이 80% 이상을 지날 경우, 앞서 설명한 바와 같이 에러를 가지면 Partial Augmentation 수행되고 학습 데이터는 업그레이드 되어 데이터세트는 점점 늘어나게 됩니다 .

<img src="https://user-images.githubusercontent.com/37643248/164343059-f2f8dc1e-22aa-4bc8-9451-031c52fde3d4.png">

### 2.3 분할 학습(Part based learning) 기반 과 레이어 셔플링(Layer Shuffling)

- 다음은 손 자세 추정을 위한 딥러닝 네트워크 구축입니다.
- 네트워크는 특징 추출 레이어단을 [hourglass](https://arxiv.org/abs/1603.06937v2)라는 검증된 네트워크를 기본으로 수정하여 사용합니다.
- 추가적으로 Part based learning과 Layer Shuffling이라는 방법을 제안합니다.
    - Part baed learning은 추출된 네트워크의 손가락 영상을 분할하여 각각 독립적인 네트워크로 학습하는 방법입니다.
    - Layer Shuffling이라는 방법은 움직임 방향이 반대인 엄지 손가락을 제외한 나머지 손가락의 레이어들이 랜덤으로 레이어가 셔플 되면서 딥러닝 학습을 수행합니다.
    - 이러한 방법은 손가락을 구분하여 세분화된 정제(Refine) 과정을 수행할 수 있으며, 비슷한 손가락들의 레이어 셔플을 통해 학습 효과를 증대시키게 됩니다.

<img src="https://user-images.githubusercontent.com/37643248/164343081-8c6c1728-68e3-4867-a4d4-f98a99820933.png">

- 모델 경량화는 기존 경량화 기능이 검증된 모듈을 활용 하였습니다.
- 표현의 편의상 MobileNet V1은 파란색 박스, MobileNet V2는 빨간색 박스,  Shufflenet V1는 초록색 박스로 정의하겠습니다.
- 네트워크에 적합한 종류와 위치에 경량화 레이어를 적용하기 위해 Grid searching 방식으로 최적화 레이어 매칭을 검증하였습니다.
- 결과적으로, 네트워크 초기 단계는 세부적인 피처를 효과적으로 뽑아 낼수 있는 Shufflenet V1을 위주로 적용하며 손가락 관절 조인트를 추출하는 단계는 인식 손실이 최소화 되는 MobileNet V1을 사용합니다. 또한 각 손가락의 파셜 레이어는 재정리를 위해 초기 MobileNet V2와 후반 MobileNet V1을 적용하였습니다.

<img src="https://user-images.githubusercontent.com/37643248/164343133-f0286d6a-a856-46be-b698-0e107e253c1c.png">

- 아래 영상은 학습과정과 구동 영상입니다. 본 논문에서 제안한 손 자세 추정 네트워크는 기존 연구된 손 자세 추정 네트워크와 비교하였고, 팜-글러브 기준으로 제안한 네트워크가 속도 및 인식률 측면에서 14.7 픽셀의 오류를 가지며 34 fps(GPU 1080Ti 기준)의 가장 좋은 결과를 도출하였습니다.
- 또한 본 CPU에서도 동작하는 모델을 개발하였고, 마지막으로 안드로이드 기반 모바일에도 적용하여 인식률은 다소 낮아졌지만 실시간으로 동작하는 21fps라는 결과를 도출하였습니다.

<iframe width="560" height="315" src="https://www.youtube.com/embed/dE9W5nAtmEY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe><br>
<iframe width="560" height="315" src="https://www.youtube.com/embed/GdpZgM-zaa8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe><br>
<img src="https://user-images.githubusercontent.com/37643248/164343139-d3d4169e-2e4e-45f0-8793-7935d141f558.png">
<iframe width="560" height="315" src="https://www.youtube.com/embed/dE9W5nAtmEY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
