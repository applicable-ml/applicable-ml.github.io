---
title: 학습한 모델 API로 만들고 배포하기
tags: ['cartoonize', 'api', 'model']
author: "Jae Heo"
---


# 학습한 모델 API로 만들고 배포하기.

![KakaoTalk_Photo_2021-10-24-12-31-54_real_A](https://user-images.githubusercontent.com/41981538/158799690-5fc0ad03-4ba3-4d94-b8e5-473f1daca63a.png)

![KakaoTalk_Photo_2021-10-24-12-31-54_fake_A](https://user-images.githubusercontent.com/41981538/158799689-ffd12f78-dd88-49d9-b79e-de6526a16a6f.png)

![Untitled](https://user-images.githubusercontent.com/41981538/158799697-a89595f8-c728-451d-b18b-bb78f582d7d8.png)

목표 독자 : 

- 딥러닝과 Backend 에 대해서, 어느정도 알지만 모델을 API 화하여, 배포하는 방법에 대해 잘 모르시는 분.

### Intro

예전에 `CartoonizeGAN` 이라는 프로젝트를 진행한 적이 있었습니다. 

이 프로젝트는 `CycleGAN`을 활용하여 현실의 사진을 웹툰 그림으로 `StyleTransfer` 하는 프로젝트입니다.

![스크린샷_2022-02-13_오전_12 14 03](https://user-images.githubusercontent.com/41981538/158799686-5ec6b57b-dfc5-489f-b9fa-dc25bbd435ac.jpg)

![스크린샷_2022-02-13_오전_12 13 56](https://user-images.githubusercontent.com/41981538/158799682-26c4d2cb-641f-4ecf-b775-922bb399dadf.jpg)

현실 이미지를 만화로 만드는 리서치가 재미있었고, 
**‘이 모델들을 언젠가 서비스화 해보고 싶다’** 라고 생각했었습니다. 

그래서, 이번 기회에 Cartoonize Gan 프로젝트에 대해서 API 화하는 작업을 진행하게 되었습니다. 

1. 모델링 
2. API 제작
3. 배포

이 과정에서 필요했던 내용들을 모아 공유드리고 싶습니다.

> API를 만들어 CPU 환경에 배포하는 내용을 다룹니다. 
GPU 환경의 배포는 개인 프로젝트의 비용 문제로 고려하지 않았습니다.
> 

관련 구현 레포

- [https://github.com/heojae/CartoonizedGanExport](https://github.com/heojae/CartoonizedGanExport)
- [https://github.com/heojae/CartoonizedGanAPI](https://github.com/heojae/CartoonizedGanAPI)
- 

---

## (Optional) 학습하기

> 본인의 도메인에 맞는 모델을 학습시키고, 모델을 추출합니다.  
이 글을 읽는 독자분의 모델(학습한 모델 혹은 학습된 모델)이 있다고 가정합니다
> 

![스크린샷_2022-03-17_오후_8 14 50](https://user-images.githubusercontent.com/41981538/158799688-3846cb91-f789-4f80-9291-dd815f3cefc8.jpg)

> [https://blog.jaysinha.me/content/images/size/w2000/2021/03/cyclegan.png](https://blog.jaysinha.me/content/images/size/w2000/2021/03/cyclegan.png)
> 

저의 경우, `CycleGAN` 을 활용하여, 학습을 진행하였습니다. 

- 원본 Repo → [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- 학습 결과 저장되어 있는 Repo → [https://github.com/heojae/CartoonizedGanExport](https://github.com/heojae/CartoonizedGanExport)

### 작업한 부분

- 원본 Repo 를 기반으로, `CycleGAN` 을 활용하여 학습함.
- Cycle Gan 을 통해서 학습되는 4가지 Model weight 중에서
    - 웹툰 Domain → 현실 Domain  :  Generator, Discriminator
    - 현실 Domain → 웹툰 Domain  :  Generator, Discriminator
      
        `현실 → 웹툰 Generator` 부분에 필요한 코드와 학습된 weight 만 빼내어 추출함.
    
- JIT 로 Model 을 Exporting 함.
- Inference Time 비교.

---

## 모델 API 화

> 학습이 완료된, 모델을 API 로 변경할 때, 고려해야하는 사항들.
(경험을 기반으로 적어 보았습니다.)
> 

아래 레포에 Flask 로 예제가 구현되어 있으며, 참고 부탁드립니다 🙏

[https://github.com/heojae/CartoonizedGanAPI](https://github.com/heojae/CartoonizedGanAPI)

### 1. Model Eval 모드로 변경하기 (Pytorch)

> `model.eval()` will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.
> 
- 해당 설정을 하지 않을 경우, `Inference` 결과가 전혀 예상하지 못하게 나오는 것을 확인할 수 있습니다.

```python
self.mode
self.model.eval()
```

참고자료

- [https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615](https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615)
- [https://blog.paperspace.com/pytorch-101-building-neural-networks/](https://blog.paperspace.com/pytorch-101-building-neural-networks/)
- [https://github.com/heojae/FoodImageRotationAdmin/issues/27#issue-806993680](https://github.com/heojae/FoodImageRotationAdmin/issues/27#issue-806993680)

### 2. no grad() 설정하기. (Pytorch)

> It will reduce memory usage and speed up computations but you won’t be able to backprop (which you don’t want in an eval script).
> 
- `Inference` 시 메모리 할당이 풀리지 않으며, 해당 과정에서 불필요한 메모리와 연산으로 인해 속도의 차이를 내게 됨.

```python
with torch.no_grad():
	output_tensor: torch.Tensor = self.model(tensor_image)  # [1, 3, 256, 256]

```

참고자료

- [https://github.com/heojae/FoodImageRotationAdmin/issues/33](https://github.com/heojae/FoodImageRotationAdmin/issues/33)
- [https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/](https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/)

### 3. [Gunicorn](https://docs.gunicorn.org/en/stable/) 설정 정하기 (WSGI)

> python 계열의 서버들은 Gunicorn 이 WSGI 로서 제공되는 경우가 많고, Gunicorn 또한 간편한 설정을 통해서, 다양한 Type 으로 실행을 할 수 있습니다.
그렇기에, Gunicorn 의 문서를 제대로 읽고, 현재 상황에 가장 적합한 설정을 정하는 것이 중요합니다.
> 

해당 부분을 고려하기 위해서는 아래 2가지를 미리 알아두면 선택의 길이 넒어집니다.

- C 단으로 내려가서 계산할 경우, GIL 의 영향을 크게 받지 않는다.
    - [https://www.youtube.com/watch?v=m2yeB94CxVQ&list=LL&index=3&t=48s](https://www.youtube.com/watch?v=m2yeB94CxVQ&list=LL&index=3&t=48s)
    - [https://discuss.pytorch.org/t/can-pytorch-by-pass-python-gil/55498/2](https://discuss.pytorch.org/t/can-pytorch-by-pass-python-gil/55498/2)
- [gunicorn 공식 문서](https://docs.gunicorn.org/en/stable/settings.html#worker-class)

> `[Locust](https://locust.io/)` 를 사용해서 부하테스트를 하여, 어떤 설정이 제일 최적인지 확인하는 과정이 있으면 좋습니다.
> 
- 개인적으로 CPU 만을 사용하는 API 에서는 `Gthread` 를 활용해서 구현하는 것을 추천.
    - 한개의 프로세스에 모델을 올리고, 가능한 Multi threading 하는 편이 더 효율적임.
    - GIL 의 영향을 많이 받지 않음. 그렇기에, Multi threading 이 생각보다 나쁘지 않음.

```bash
gunicorn -k gthread --workers=2 --threads=4 --bind 0.0.0.0:8080 wsgi
```

- Batch 처리에서는 `Sync` 를 통해서 한개의 프로세스 만으로 활용하기를 추천

```bash
gunicorn -k sync --workers=1 --threads=1 --bind 0.0.0.0:8080 wsgi
```

### 4. PyTorch - CPU version

> PyTorch 의 경우, GPU version 과 CPU version 의 경우를 나누어서 설정을 할 필요가 있습니다.
 해당 설정만으로도, 1개의 프로세스에 2GB 이상의 차이가 나타날 수 있고,
 GPU 에서 올릴 일이 없다면, CPU version 을 설치하는 것이 필요로 합니다.
> 

![Untitled](%E1%84%92%E1%85%A1%E1%86%A8%E1%84%89%E1%85%B3%E1%86%B8%E1%84%92%E1%85%A1%E1%86%AB%20%20ffac4/Untitled%201.png)

아래와 같은 설정의 차이 많으로도, 메모리 사용량에서 큰 차이가 날 수 있음

- CPU Version library

```bash
pip3 install --no-cache-dir torch==1.10.0+cpu -f <https://download.pytorch.org/whl/torch_stable.html>
pip3 install --no-cache-dir torchvision==0.11.0+cpu -f <https://download.pytorch.org/whl/torch_stable.html>

```

### 5. (Optional) JIT 로 Exporting 하기

> 배포 과정에서 필수적인 부분은 아니지만, 프로덕션 환경에서 코드정리 부분의 이점이 있습니다.
> 
- CPU 환경 위에서는 큰 발전은 없지만, GPU 환경에서는 Inference 가 빨라지기 때문에 사용하는 것을 추천
- Torch 에서 문제가 되는 `네트워크 정의`  없이 사용할 수 있으며, 단순히 Model 만 교체하면 해결됨

```python
model = torch.jit.load(self.jit_path)
```

참고자료

- [https://towardsdatascience.com/pytorch-jit-and-torchscript-c2a77bac0fff](https://towardsdatascience.com/pytorch-jit-and-torchscript-c2a77bac0fff)

---

## AWS EB 를 활용해서, API 배포하기

![%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-01-12_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_8 10 48](https://user-images.githubusercontent.com/41981538/158799679-a5e7265c-93d6-4a52-af77-a02699fb211b.jpg)

**EB (Elastic Beanstalk) 를 사용하는 이유**

- MLOps 관점에서 [k8s](https://kubernetes.io/ko/)를 직접 구성해서 사용할수도 있으나, 
부족한 리소스 안에서 방대한 [k8s](https://kubernetes.io/ko/)를 공부하고 구축하고 효율적으로 적용은  현실적으로 쉽지 않을수 있음
- EB란 AWS에서 제공하는 로드벨런싱 지원 제품 중 하나로, 
Backend 개발자 한명이 AWS 환경에서 쉽게 로드벨런싱을 지원하여 대용량 트래픽을 Scale Out하게 처리할 수 있는 이점이 있음

아래 단계들이 필요로 합니다. 

### 1. AWS IAM 계정 생성하기

- 그룹 및 사용자 생성
- `Administrator Access - AWS Elastic Beanstalk` 추가
- `AmazonEC2ContainerRegistryFullAccess` 추가

참고자료

- [https://docs.aws.amazon.com/ko_kr/IAM/latest/UserGuide/introduction.html](https://docs.aws.amazon.com/ko_kr/IAM/latest/UserGuide/introduction.html)
- [https://tech.cloud.nongshim.co.kr/2018/10/13/초보자를-위한-aws-웹구축-2-iam-유저-생성하기/](https://tech.cloud.nongshim.co.kr/2018/10/13/%EC%B4%88%EB%B3%B4%EC%9E%90%EB%A5%BC-%EC%9C%84%ED%95%9C-aws-%EC%9B%B9%EA%B5%AC%EC%B6%95-2-iam-%EC%9C%A0%EC%A0%80-%EC%83%9D%EC%84%B1%ED%95%98%EA%B8%B0/)

### 2. AWS ECR 에 Docker Image 올리기

- ECR 에 들어가서, Priviate or Public Repository 생성
- Docker Image 올리기.

```bash
# public 으로 활용할 경우입니다. 
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/{your ecr id}

docker build -t {your ecr repo name} .

docker tag cartoonize_api:latest public.ecr.aws/{your ecr id}/{your ecr repo name}:latest

docker push public.ecr.aws/{your ecr id}/{your ecr repo name}:latest
```

참고자료

- [https://docs.aws.amazon.com/ko_kr/AmazonECR/latest/userguide/docker-push-ecr-image.html](https://docs.aws.amazon.com/ko_kr/AmazonECR/latest/userguide/docker-push-ecr-image.html)
- [https://aws.amazon.com/ko/ecr/pricing/](https://aws.amazon.com/ko/ecr/pricing/)

### 3. EB 에 배포하기. (실험 포함)

- 환경 티어 - 웹 서버 환경 생성.
  
    ![%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-01-02_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11 16 39](https://user-images.githubusercontent.com/41981538/158799655-de920919-bbf5-4b95-b63f-ff93ad43c05d.jpg)
    
- 플랫폼 - Docker 으로 설정
  
    ![%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-01-02_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11 16 57](https://user-images.githubusercontent.com/41981538/158799668-1b27d325-872b-4135-80f9-ed58836b9f85.jpg)
    
- 환경 구성 - 프리티어를 사용해서 진행.
  
    ![스크린샷_2022-02-13_오전_1 43 12](https://user-images.githubusercontent.com/41981538/158799680-b2d904f2-c4e3-4dc7-b967-4a613fd5f0e6.jpg)
    
    
    
- 환경 구성 - 추후 비용을 들여서 배포가 필요로 할 경우에는, 고가용성을 설정하여, 로드 밸런서 기능을 추가할 필요가 있음.
  
    ![스크린샷_2022-02-13_오전_1 43 12](https://user-images.githubusercontent.com/41981538/158799680-b2d904f2-c4e3-4dc7-b967-4a613fd5f0e6.jpg)
    
    ![%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-01-02_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11 30 22](https://user-images.githubusercontent.com/41981538/158799677-220fb789-714e-488d-8282-151145c413ab.jpg)
    
- 애플리케이션 코드 - `[Dockerrun.aws.json](https://github.com/heojae/CartoonizedGanAPI/blob/main/Dockerrun.aws.json)` 을 통해서, 관리.
  
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
    
    ![Untitled 3](https://user-images.githubusercontent.com/41981538/158799696-e2ebb85e-3f2a-4b33-a0e1-dbe999914191.png)
    

설정을 완료하면, 아래와 같은 화면이 뜨면서, 해당 API 가 올라가는 것을 볼 수 있음.

![스크린샷_2022-02-13_오전_2 04 22](https://user-images.githubusercontent.com/41981538/158799681-7406e133-57e5-40d7-8c7c-3a3a0b009e23.jpg)

**부하 테스트 실험 내용.**

아래 파일을 로컬에서 파이썬으로 실행시켜서, EB 로 Request 로 보냄.

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

![Untitled](https://user-images.githubusercontent.com/41981538/158799697-a89595f8-c728-451d-b18b-bb78f582d7d8.png)

### 결과

200개 의 Request 를 동시에 각각 보냄.

- 1 개 client → 평균 : 1.82s
- 2개 client → 평균 : 3.64s

`[t2.micro](https://aws.amazon.com/ko/ec2/instance-types/t2/)` 와 같이 CPU 성능이 높지 않은 곳에서 딥러닝 모델을 돌리는 것이 좋지 않은 결과를 나타낸다는 것을 알고 있었으나, 
이정도 Request 밖에 되지 않는데 `CPU 사용률` 그래프를 보니 안타까움.

해결방법

사실 아래 2개를 하면 되는 일이긴 하고, EB 에서 원래 지원해주는 기능이기에 별일은 아님.

- EC2 인스턴스 설정에서 부터, CPU 가 좋은 인스턴스를 활용.
- Load Balancing 을 활용하여, 여러 인스턴스를 올림.

- 참조 링크
    - [https://docs.aws.amazon.com/ko_kr/elasticbeanstalk/latest/dg/create_deploy_docker.html](https://docs.aws.amazon.com/ko_kr/elasticbeanstalk/latest/dg/create_deploy_docker.html)
    - [https://docs.aws.amazon.com/ko_kr/elasticbeanstalk/latest/dg/single-container-docker-configuration.html](https://docs.aws.amazon.com/ko_kr/elasticbeanstalk/latest/dg/single-container-docker-configuration.html)
    - [https://medium.com/devops-with-valentine/how-to-deploy-a-docker-container-to-aws-elastic-beanstalk-using-aws-cli-87ccef0d5189](https://medium.com/devops-with-valentine/how-to-deploy-a-docker-container-to-aws-elastic-beanstalk-using-aws-cli-87ccef0d5189)

---

### Summary

모델을 API 화하고, 배포하는 과정에 대해서 정리를 해보았습니다.  

제 경험을 바탕으로 작성하였고 광범위한 내용을 다루긴 하지만, 
여러분의 AI 프로젝트를 서비스화 하는데 도움이 되었으면 좋겠습니다. 🙏
