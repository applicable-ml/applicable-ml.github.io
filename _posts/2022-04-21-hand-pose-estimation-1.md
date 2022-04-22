---
title: Hand Gesture/Pose Estimation - 1 (Image Processing)
tags: ['ml', 'vision', 'hand-pose-estimation', 'pose-estimation', 'preprocessing']
author: "Dongseok Yang"
---

## 0. 시작하기

- 해당 포스팅의 목적은 “팜 글러브: 시계형 적외선 카메라 기반 손 HCI 하드웨어 및 소프트웨어 제작” 입니다.
- 이번 포스팅은 3부분으로 구성되어 있습니다.
- 첫번째 포스팅
    - **Hand Gesture/Pose Estimation - 1 (Image Processing)**
    - HW 구성, 순수 영상처리 기법을 활용한 손 자세 추정 내용으로 구성되어 있습니다.
- 두번째 포스팅
    - **Hand Gesture/Pose Estimation - 2 (Deep Learning)**
    - 손 자세 추정을 위해 딥러닝 접근법을 설명합니다. 또한 모델 경량화를 위한 내용이 포함되어 있습니다.
- 세번째 포스팅
    - **Hand Gesture/Pose Estimation - 3 (Gesture Recognition)**
    - 해당 포스팅은 Thumb-to Finger Tap이라는 제스처를 인식하기 위한 기술을 설명합니다.


## 1. Intro

- 해당 내용은 손 영상을 기반한 인터페이스 기술에 대한 내용입니다.
- HW 구성과 영상처리 기법을 활용하여 손가락의 운동학적 특징을 추출하여 인터페이스의 명령 매체로 활용하기 위한 기술 내용입니다.

## 2. Overview

- 최근 기술의 발전으로 인간의 작업보다 쉽게 수행 할 수 있도록 컴퓨팅 시스템이 설계되었습니다. 이로 인해 HCI(Human Computer Interface) 즉, 컴퓨터와 인간의 상호작용은 우리 삶의 중요한 부분이 되었습니다.
- 또한 가까운 미래에 컴퓨팅, 통신 및 디스플레이에 사용할 수 있는 기존 인터페이스 기술 (키보드, 마우스)들에 병목 될 것이며, 가상현실 및 혼합현실 등 변화하는 환경에 자연스럽게 접목될 인터페이스가 필요합니다.
- 그 중 손은 의사소통을 하는 과정에서 가장 많이 활용되는 신체 입니다. 따라서 손은 HCI 분야에 핵심적인 관심 분야입니다.
- 본 기사의 목표는 팜 글러브라는 디바이스와 영상처리 기법을 위해 재활 환자들게 디지털 손 재활 훈련을 제공하기 위한 목적을 가지고 있습니다.

<img src="https://user-images.githubusercontent.com/37643248/164342167-8a2d9a3d-d6d9-43b3-8b6a-fcf07a9e89f8.png">

<img src="https://user-images.githubusercontent.com/37643248/164342190-b51d7f70-3fd5-45cf-820d-002e32ccd0d9.png">

## 3. Hardware

- 해당 목적은 인식 알고리즘에 중점을 맞추기었기 때문에 HW의 경우 동향과 제안한 방법에 대해서만 간단히 언급하겠습니다.
- 이전 연구에서 손을 분석하기 위한 하드웨어 장치로서, [Flex 센서](https://en.wikipedia.org/wiki/Flex_sensor), [IMU 센서](https://towardsdatascience.com/what-is-imu-9565e55b44c) 등을 적용하여 손을 인터페이스로 활용하는 시도를 해왔습니다
- 그러나, 이러한 센서들은 모든 손가락의 정확한 위치에 부착되어야 하며 착용성에 불편함을 가집니다. 또한 커버형으로 구성되기 때문에 손가락의 촉감을 완전히 잃게 됩니다.
- 이러한 문제점은 카메라를 사용하면 해결됩니다.
- 하지만 거치형 카메라는 공간에 제약을 받고, 시계형 타입 카메라 디바이스는 손목의 움직임에 따라 손의 뷰를 얻는데 제한적 입니다.

<img src="https://user-images.githubusercontent.com/37643248/164342222-9920b26c-0911-4a98-ab16-e0bbc5733cca.png">

- 제안한 손 영상 획득을 위한 HW 구성은 손바닥에 부착하는 카메라로 손 자세 및 동작 분석이 가능한 웨어러블 디바이스 입니다. 또한 블루투스 및 HID 모듈이 내장되어 있어 피시 및 스마트 기기와 인터페이스 연동이 가능합니다.
- 이러한 구조의 장점은 다음과 같습니다.

---

> **첫번째 팜 글러브는 손목 움직임에 제한을 받지 않습니다.**
> 
1. 기존 손목 기반 카메라 디바이스는 손목을 뒤로 할 경우 손 및 손가락의 영상을 잃을 수 있습니다.
2. 하지만 오른쪽과 같이 팜-글러브는 손바닥에 부착하기 때문에 안정적인 손 영상을 얻을 수 있습니다.

> **두번째 손가락의 일정한 방향성 입니다.**
> 
1. 팜-글러브에서 들어오는 손 영상에서 엄지손가락은 오른쪽, 나머지 손가락은 왼쪽으로 방향성을 가집니다. 이러한 장점은 머신러닝 및 영상처리 관점에서 매우 가치있는 특징 추출이 가능합니다.

> **마지막으로 정확한 손 객체 획득, 즉 노이즈 처리가 용이합니다.**
> 
1. 외부 배경에 대해서 손 객체를 추출하는 것은 왼쪽 그림과 같이 매우 어려운 과제 입니다.
2. 하지만 팜 –글러브에서는 일상 생활에 활용되지 않은 940nm 적외선 송수신 카메라를 사용하기 때문에 간단한 전처리만으로도 쉽게 손 객체를 얻을 수 있습니다.

---

<img src="https://user-images.githubusercontent.com/37643248/164342236-659293a0-a480-4e75-a4eb-8ae35aeb3236.png">

## 4. 영상처리 - 손의 운동학적 분석 기반 기하학적 특징 추출

- 본격적인 Hand gesture/pose estimation에 대한 알고리즘을 분석해 보겠습니다. 본 단락에서는 순수한 영상처리 기법만을 활용하여 손의 운동학적 특징을 추출하는 알고리즘에 대해서 소개합니다.
- 또한 손가락이 폐색되어 보이지 않는 경우 손의 관절 부분인 MCP 라인 추출 기법을 손가락의 인덱스를 정확히 판단하는 기법에 대해서도 소개합니다.
- 먼저 손 객체 추출에 대해서 말씀 드리겠습니다.
- 앞서 설명드렸듯이, 팜 글러브에서는 HSV 컬러모델의 Value 값의 이진화 분리만을 가지고도 견고한 손 객체 추출이 가능합니다.

<img src="https://user-images.githubusercontent.com/37643248/164342258-5117efd7-6ffb-487d-bad2-51c41266f8aa.png">

- 손 객체를 얻으면 손의 상단 경계선을 추출합니다. 방식은 순차적 서칭 방식으로 사용하며 픽셀 값이 0이되지 않는 구간의 모든 가로 점으로 추출합니다.
- 또한 추출된 점은 이전 점과 비교하여 상승 점인지 하강 점인지 구분되며 상승과 하강이 교차하는 점을 그림의 보라색과 같이 손가락 끝점으로 정의합니다.

<img src="https://user-images.githubusercontent.com/37643248/164342281-2e48f8c6-ab83-4175-9f10-8de39067dbb9.png">

- 하지만 보시는 그림과 같이, 엄지손가락에서 상승 하강 곡선이 생성되지 않아 손가락 끝점이 추출되지 않은 경우가 할 수도 있습니다.

<img src="https://user-images.githubusercontent.com/37643248/164342295-fd12c76e-19dd-4389-a71c-93b2ff8a5bc3.png">

- 따라서 명확한 손가락 끝점을 찾기 위해 오른쪽 그림의 노란점과 같이 일정한 오프셋을 가지는 4가지 후보군 극점을 추가합니다.
- 이후 4가지 후보군 극점과 추출된 손가락 끝점을 기준으로 우하-방향성 마스크를 이용하여 손가락 경계선 추적 알고리즘이 적용됩니다
- 우향-방향성 마스크는 엄지를 제외한 다른 손가락의 외각선을 정확히 추출할수 있습니다.
- 또한 그림에 보라색 원과 같이 일정한 높이 기준으로 사영되는 점을 사영점이라고 정의합니다.
- 이로 인해, 추출된 외각선을 기반과 사영점을 활용하여 손가락 움직임의 세부 분석이 가능합니다.

<img src="https://user-images.githubusercontent.com/37643248/164342321-edd8c73a-74f9-4df0-82e3-a39fd1e6448b.png">

- 마지막으로 [MCP(**Metacarpophalangeal](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=spm0808&logNo=40207380223))** 라인을 이용한 손가락 인덱싱 방법에 대해서 말씀드리겠습니다.
- MCP 라인을 구하는 방법은 왼쪽 위의 그림과 같습니다.
- 빨간색 점은 이전 단계에서의 손의 상단 경계선이며, 손가락 끝을 찾는 방법과 반대로 하강-상승하는 포인트를 기준점으로 MCP라인을 연결합니다.
- 팜-글러브의 특성상 오른쪽 위 그림과 같이 손가락이 존재하는 범위가 고정되어 있습니다.
- 실험결과 위와 같이 손가락이 존재하는 범위를 정의하였고 MCP라인 기준으로 손가락이 존재하지 않는다면 손가락 인덱싱 과정에서 제외됩니다.
- 아래 그림들에서 손가락이 없어진 상황에서도 정확한 인덱싱 결과를 도출하는 것을 보실 수가 있습니다.

<img src="https://user-images.githubusercontent.com/37643248/164342342-95b90727-5f46-4065-a490-371aa449e80e.png">

<iframe width="560" height="315" src="https://www.youtube.com/embed/UtzxFnwHw4g" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Reference

- Karam M (2006) A framework for research and design of gesture-based human computer interactions. PhD Thesis, University of Southampton
- Loclair, C., Gustafson S., and Baudisch P.: ‘PinchWatch: a wearable device for one-handed microinteractions’, *Proc. Mobile HCI*., 2010,
- Huang, M-C., et al.: ‘Smartglove for upper extremities rehabilitative gaming assessment’, *Proc. 5th Int. Conf. Pervasive Technol. Assistive Environ*., ACM, 2012.
- Valtin, M., et al.: ‘Modular finger and hand motion capturing system based on inertial and magnetic sensors’, *Curr. Directions Biomed. Eng.*, 2017, 3.1 , pp. 19–23.
- Mehring, C., et al.: ‘KITTY: Keyboard independent touch typing in VR’, *Virtual Reality Proc. IEEE*., 2004
- Way, D., : ‘A usability user study concerning free-hand microgesture and wrist-worn sensors’, Wearable and Implantable Body Sensor Networks (BSN), 2014 *11th Int. Conf. IEEE*, 2014
- Pratorius, M., et al.: ‘Sensing Thumb-to-Finger Taps for Symbolic Input in VR/AR Environments’, *IEEE Comp. Graphics Appl.*, 2015

## 용어 정리

**손의 운동학정 특징 :** 일반적으로 인간 손에 있는 검지 손가락의 평면운동은 3개의 관절운동에 의해 이루어진다. 이러한 운동을 위해서는 기본적으로 역기국학 문제로 접근되어 햔다. 손을 활용한 파지나, 조작행위에 있어서 필수적인 운동 정의를 뜻한다.

 **[MCP(Metacarpophalangeal):](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=spm0808&logNo=40207380223) MCP 관절이라고도 하며 손등과 연결되는 관절을 말한다. 중수지절간관절이라고 한다. 말로표현하기 어렵지만 링크의 그림을 확인하면 쉽게 파악할수 있다.**
