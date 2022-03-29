---
title: 학습된 GAN 모델로 AWS 기반의 API 서버 제작, 배포, 테스트, 그리고 확장가능한 구조 만들기
tags: ['cartoonize', 'api', 'model', 'aws', 'es', 'model-serving', 'deployment']
author: "Jae Heo"
---

<img width="1105" alt="image" src="https://user-images.githubusercontent.com/37643248/160271633-e4da7fd8-8c57-4a0e-94a6-b59f4350d046.png">

> 목표 독자: 딥러닝과 백엔드에대한 기초 지식이 있지만, 모델 서빙을 위해 API 서버 앱을 제작, 배포, 테스트하는 전체적인 과정이 궁금하신 분

### Intro

예전에 CartoonizeGAN 이라는 프로젝트를 진행한 적이 있었습니다. 이 프로젝트는 [CycleGAN](https://arxiv.org/abs/1703.10593)을 활용하여 현실의 사진을 웹툰 그림으로 [StyleTransfer](https://paperswithcode.com/task/style-transfer) 하는 프로젝트입니다.

<img width="456" alt="Screen Shot 2022-03-27 at 11 19 29 AM" src="https://user-images.githubusercontent.com/37643248/160263851-9763ac81-c886-4b81-b175-bf29d120bfe2.png">

현실 이미지를 만화로 만드는 리서치가 재미있었고, 
**‘이 모델들을 언젠가 서비스화 해보고 싶다’** 라고 생각했었습니다. 

그래서, 이번 기회에 CartoonizeGan 프로젝트에 대해서 API 화하는 작업을 진행하게 되었습니다. 

1. GAN 모델 제작
2. API 서버앱 제작
3. 배포 및 부하테스트

아래 과정에서 필요했던 내용들을 모아 공유드리고 싶습니다.

> API를 만들어 CPU 환경에 배포하는 내용을 다룹니다. 
GPU 환경의 배포는 개인 프로젝트의 비용 문제로 고려하지 않았습니다.

구현 저장소

- [https://github.com/heojae/CartoonizedGanExport](https://github.com/heojae/CartoonizedGanExport)
- [https://github.com/heojae/CartoonizedGanAPI](https://github.com/heojae/CartoonizedGanAPI)

---

## (Optional) GAN 모델 학습하기

> 본인의 도메인에 맞는 모델을 학습시키고, 모델을 추출합니다.  
이 글을 읽는 독자분의 모델(학습한 모델 혹은 학습된 모델)이 있다고 가정합니다

<img src="https://user-images.githubusercontent.com/37643248/160269072-698ec368-d11f-4bb7-964b-43d3aa07a8fd.jpeg" width=700px height=400px>

> [https://blog.jaysinha.me/content/images/size/w2000/2021/03/cyclegan.png](https://blog.jaysinha.me/content/images/size/w2000/2021/03/cyclegan.png)

저의 경우, CycleGAN을 활용하여, 학습을 진행하였습니다. 

- 글쓴이가 참고한 저장소 → [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- 직접 학습 결과 저장되어 있는 저장소 → [https://github.com/heojae/CartoonizedGanExport](https://github.com/heojae/CartoonizedGanExport)

### 작업 내용

- 원본 저장소를 기반으로, CycleGAN을 활용하여 학습하였습니다
- Cycle Gan을 통해서 학습되는 4가지 모델 weight 중에서
    - 웹툰 Domain → 현실 Domain  :  Generator, Discriminator
    - 현실 Domain → 웹툰 Domain  :  Generator, Discriminator
        **현실 사진 이미지를 웹툰 이미지로 생성하는 부분**에 필요한 코드와 학습된 weight만 가져옴
- JIT 로 Model 을 Exporting 함
- Inference Time 비교

---

## 모델 API 화

> 학습이 완료된, 모델을 API 로 변경할 때, 고려해야하는 사항들.
(경험을 기반으로 적어 보았습니다.)

아래 레포에 Flask 로 예제가 구현되어 있으며, 참고 부탁드립니다 🙏

[https://github.com/heojae/CartoonizedGanAPI](https://github.com/heojae/CartoonizedGanAPI)

### 1. 모델 Eval 모드로 변경하여 추론하기 (Pytorch)

> `model.eval()` will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.
> 
- 해당 설정을 하지 않을 경우, 추론 결과가 전혀 예상하지 못하게 나오는 것을 확인할 수 있습니다.

```python
self.model = JitCycleGanModel()
self.model.eval()
```

### 2. `no_grad()` 설정하기. (Pytorch)

> It will reduce memory usage and speed up computations but you won’t be able to backprop (which you don’t want in an eval script).

- `Inference` 시 메모리 할당이 풀리지 않으며, 해당 과정에서 불필요한 메모리와 연산으로 인해 속도의 차이를 내게 됨.

```python
with torch.no_grad():
	output_tensor: torch.Tensor = self.model(tensor_image)  # [1, 3, 256, 256]

```

### 3. [Gunicorn](https://docs.gunicorn.org/en/stable/) 설정 정하기 (WSGI)

> python 계열의 서버들은 Gunicorn 이 WSGI 로서 제공되는 경우가 많고, Gunicorn 또한 간편한 설정을 통해서, 다양한 Type 으로 실행을 할 수 있습니다.
그렇기에, Gunicorn 의 문서를 제대로 읽고, 현재 상황에 가장 적합한 설정을 정하는 것이 중요합니다.

해당 부분을 고려하기 위해서는 아래 2가지를 미리 알아두면 선택의 길이 넒어집니다.

- C언어 레벨로 내려가서 계산할 경우, [GIL](https://en.wikipedia.org/wiki/Global_interpreter_lock)의 영향을 크게 받지 않음
    - [https://www.youtube.com/watch?v=m2yeB94CxVQ&list=LL&index=3&t=48s](https://www.youtube.com/watch?v=m2yeB94CxVQ&list=LL&index=3&t=48s)
    - [https://discuss.pytorch.org/t/can-pytorch-by-pass-python-gil/55498/2](https://discuss.pytorch.org/t/can-pytorch-by-pass-python-gil/55498/2)
- [gunicorn 공식 문서](https://docs.gunicorn.org/en/stable/settings.html#worker-class)

> [Locust](https://locust.io/)를 사용해서 부하테스트를 하여, 어떤 설정이 제일 최적인지 확인하는 과정이 있으면 좋습니다.

- CPU만을 사용하는 서버 앱에서는 경험적으로 Gthread가 좋은 성능을 냄
    - GIL의 영향을 많이 받지 않아서인지 한개의 프로세스에 모델을 올리고, 가능한 Multi threading 하는 것이 여러 프로세스를 사용하는것보다 더 좋은 성능을 냄

```bash
gunicorn -k gthread --workers=2 --threads=4 --bind 0.0.0.0:8080 wsgi
```

- Batch 처리에서는 마찬가지 경험적으로 Sync 를 통해서 한개의 프로세스 만으로 활용하는 것이 좋은 성능을 냄

```bash
gunicorn -k sync --workers=1 --threads=1 --bind 0.0.0.0:8080 wsgi
```

### 4. CPU 사용 Pytorch 설치하기

> PyTorch 의 경우, GPU version 과 CPU version 의 경우를 나누어서 설정을 할 필요가 있습니다.
 해당 설정만으로도, 1개의 프로세스에 2GB 이상의 차이가 나타날 수 있고,
 GPU 에서 올릴 일이 없다면, CPU version 을 설치하는 것이 필요로 합니다.

<img src="https://user-images.githubusercontent.com/41981538/158799691-3c2a4cbd-8a5c-4d3b-9049-9552a3275972.png" width=800px height=70px>

아래와 같은 설정의 차이 많으로도, 메모리 사용량에서 큰 차이가 날 수 있음

```bash
# cpu만 사용
pip3 install --no-cache-dir torch==1.10.0+cpu torchvision==0.11.0+cpu -f <https://download.pytorch.org/whl/torch_stable.html>
# cuda 사용
pip3 install --no-cache-dir torch==1.10.0+cu102 torchvision==0.11.0+cu102 -f <https://download.pytorch.org/whl/torch_stable.html>
```

### 5. (Optional) JIT 로 Exporting 하기

> 배포 과정에서 필수적인 부분은 아니지만, 프로덕션 환경에서 코드정리 부분의 이점이 있습니다.

- 추론시 CPU 환경에서는 큰 성능향상은 없지만, GPU 환경에서는 추론 속도에 이점이 있음
- 네트워크 정의 구현부 없이 모델을 불러오고 추론할 수 있으므로 유지보수에 용이

```python
model = torch.jit.load(self.jit_path)
```

### 참고자료

- [https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615](https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615)
- [https://blog.paperspace.com/pytorch-101-building-neural-networks/](https://blog.paperspace.com/pytorch-101-building-neural-networks/)
- [https://github.com/heojae/FoodImageRotationAdmin/issues/27#issue-806993680](https://github.com/heojae/FoodImageRotationAdmin/issues/27#issue-806993680)
- [https://github.com/heojae/FoodImageRotationAdmin/issues/33](https://github.com/heojae/FoodImageRotationAdmin/issues/33)
- [https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/](https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/)
- [https://towardsdatascience.com/pytorch-jit-and-torchscript-c2a77bac0fff](https://towardsdatascience.com/pytorch-jit-and-torchscript-c2a77bac0fff)

---

## AWS EB 를 활용해서, API 배포하기

<img src="https://user-images.githubusercontent.com/41981538/158799679-a5e7265c-93d6-4a52-af77-a02699fb211b.jpg" width=500px height=200px>

### EB(Elastic Beanstalk) 를 사용하는 이유

- MLOps 관점에서 [k8s(kubernates)](https://kubernetes.io/ko/)를 직접 구성해서 사용할수도 있으나, 부족한 리소스 안에서 방대한 k8s를 공부하고 구축하고 효율적으로 적용은 현실적으로 쉽지 않을수 있음
- [EB](https://aws.amazon.com/ko/elasticbeanstalk/?nc1=h_ls)란 AWS에서 제공하는 [부하분산(로드벨런싱)](https://ko.wikipedia.org/wiki/%EB%B6%80%ED%95%98%EB%B6%84%EC%82%B0) 지원 제품 중 하나로, AWS 환경에서 백엔드 개발자가 쉽게 부하분산을 지원하여 대용량 트래픽을 스케일-아웃할 수 있는 시스템 구축이 가능하도록 도움을 줌
아래 단계들이 필요로 합니다. 

### 1. AWS IAM 계정 생성하기

- 그룹 및 사용자 생성
- `Administrator Access - AWS Elastic Beanstalk` 추가
- `AmazonEC2ContainerRegistryFullAccess` 추가

### 2. AWS ECR 에 Docker Image 올리기

- [ECR](https://aws.amazon.com/ko/ecr/)에 들어가서, Priviate or Public Repository 생성
- Docker Image 올리기

```bash
# public 으로 활용할 경우입니다. 
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/{your ecr id}

docker build -t {your ecr repo name} .

docker tag cartoonize_api:latest public.ecr.aws/{your ecr id}/{your ecr repo name}:latest

docker push public.ecr.aws/{your ecr id}/{your ecr repo name}:latest
```

### 3. EB 에 배포하기 (실험 포함)

- 환경 티어 - 웹 서버 환경 생성<br>
    <img src="https://user-images.githubusercontent.com/41981538/158799655-de920919-bbf5-4b95-b63f-ff93ad43c05d.jpg" width=500px>
- 플랫폼 - Docker 으로 설정<br>
    <img src="https://user-images.githubusercontent.com/41981538/158799668-1b27d325-872b-4135-80f9-ed58836b9f85.jpg" width=500px>
- 환경 구성 - 프리티어를 사용해서 진행<br>
    <img src="https://user-images.githubusercontent.com/41981538/158799680-b2d904f2-c4e3-4dc7-b967-4a613fd5f0e6.jpg" width=500px>
- 환경 구성 - 추후 비용을 들여서 배포가 필요로 할 경우에는, 고가용성을 설정하여, 부하분산 기능을 추가할 필요가 있음<br>
    <img src="https://user-images.githubusercontent.com/41981538/158799680-b2d904f2-c4e3-4dc7-b967-4a613fd5f0e6.jpg" width=500px><br>
    <img src="https://user-images.githubusercontent.com/41981538/158799677-220fb789-714e-488d-8282-151145c413ab.jpg" width=500px>
- 애플리케이션 코드 - [`Dockerrun.aws.json`](https://github.com/heojae/CartoonizedGanAPI/blob/main/Dockerrun.aws.json)을 통해서, 관리
  
    ```json
    # Dockerrun.aws.json
    {
      "AWSEBDockerrunVersion": "1",
      "Image": {
        "Name": "your ecr repo docker image url"
      },
      "Ports": [
        {
          "ContainerPort": 8080
        }
      ],
      "Volumes": [
        {
          "HostDirectory": "/",
          "ContainerDirectory": "/"
        }
      ]
    }
    
    ```
    
    <img src="https://user-images.githubusercontent.com/41981538/158799696-e2ebb85e-3f2a-4b33-a0e1-dbe999914191.png" width=700px height=200px>

설정을 완료하면 아래와 같은 화면이 나타나며 해당 API 서버가 실행되는 것을 볼 수 있습니다.

<img src="https://user-images.githubusercontent.com/41981538/158799681-7406e133-57e5-40d7-8c7c-3a3a0b009e23.jpg" width=700px>

### 부하 테스트 실험

로컬에서 파이썬으로 아래 스크립트를 실행시켜 EB로 Request 로 보냈습니다.

```python
import requests
import time

req_times = []
url = f'{eb 에서 제공해주는 url}/cartoonize'
all_count = 200
files = {'image': open('./sample/a.png', 'rb')}
for i in range(all_count):
    start_time = time.time()
    response = requests.post(url, files=files)
    end_time = time.time()
    print("------------------------ Response 받기 완료 ------------------------")
    print(response.status_code)
		print(i, end_time - start_time)
    req_times.append(end_time - start_time)

print("평균 : ", sum(req_times)/all_count)**
```

<img src="https://user-images.githubusercontent.com/41981538/158799697-a89595f8-c728-451d-b18b-bb78f582d7d8.png" width=700px height=400px>

### 결과

200개 의 Request를 동시에 각각 보냈습니다.

- Client 1개 → 평균 응답시간: 1.82s
- Client 2개 → 평균 응답시간: 3.64s

글쓴이가 사용한 추론 장비는 [t2.micro](https://aws.amazon.com/ko/ec2/instance-types/t2/)인데, CPU 성능이 좋지 않은 장비에서 딥러닝 모델을 실행하였습니다. 위 그래프를 통해 알 수 있듯, 적은 요청량으로 CPU 사용율이 100%를 달성해버린 점이 아쉽네요.
#### 해결방법

이러한 문제는 아래의 EB 자체 제공 기능들을 활용하여 손쉽게 해결할 수 있습니다.

- CPU가 좋은 EC2 인스턴스를 사용
- 부하분산을 활용하여 여러 인스턴스를 올림

### 참조 링크

- [https://docs.aws.amazon.com/ko_kr/IAM/latest/UserGuide/introduction.html](https://docs.aws.amazon.com/ko_kr/IAM/latest/UserGuide/introduction.html)
- [https://tech.cloud.nongshim.co.kr/2018/10/13/초보자를-위한-aws-웹구축-2-iam-유저-생성하기/](https://tech.cloud.nongshim.co.kr/2018/10/13/%EC%B4%88%EB%B3%B4%EC%9E%90%EB%A5%BC-%EC%9C%84%ED%95%9C-aws-%EC%9B%B9%EA%B5%AC%EC%B6%95-2-iam-%EC%9C%A0%EC%A0%80-%EC%83%9D%EC%84%B1%ED%95%98%EA%B8%B0/)
- [https://docs.aws.amazon.com/ko_kr/AmazonECR/latest/userguide/docker-push-ecr-image.html](https://docs.aws.amazon.com/ko_kr/AmazonECR/latest/userguide/docker-push-ecr-image.html)
- [https://aws.amazon.com/ko/ecr/pricing/](https://aws.amazon.com/ko/ecr/pricing/)
- [https://docs.aws.amazon.com/ko_kr/elasticbeanstalk/latest/dg/create_deploy_docker.html](https://docs.aws.amazon.com/ko_kr/elasticbeanstalk/latest/dg/create_deploy_docker.html)
- [https://docs.aws.amazon.com/ko_kr/elasticbeanstalk/latest/dg/single-container-docker-configuration.html](https://docs.aws.amazon.com/ko_kr/elasticbeanstalk/latest/dg/single-container-docker-configuration.html)
- [https://medium.com/devops-with-valentine/how-to-deploy-a-docker-container-to-aws-elastic-beanstalk-using-aws-cli-87ccef0d5189](https://medium.com/devops-with-valentine/how-to-deploy-a-docker-container-to-aws-elastic-beanstalk-using-aws-cli-87ccef0d5189)

---

### Summary

모델을 API 화하고, 확장가능한 구조로 배포 및 테스트 한 내용에 대해서 정리를 해보았습니다. 제 경험을 바탕으로 작성하였고 광범위한 내용을 다루긴 하지만, 여러분의 AI 프로젝트를 서비스화 하는데 도움이 되었으면 좋겠습니다. 🙏