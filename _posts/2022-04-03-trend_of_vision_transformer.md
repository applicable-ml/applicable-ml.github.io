---
title: Trend of Vision Transformer
tags: ['ml', 'vision', 'transformer', 'lightweight', 'tiny']
author: "Sangbum Choi"
---

# Vision-Transformer

Created: November 12, 2021 10:56 AM
Tags: APP-ML

### Introduction

이 포스트같은 경우 Transformer가 원래는 NLP 도메인에서 주로 연구된 아키텍쳐인데, 최근에는 Vision쪽에서 도입을 시도하여 어느정도의 성과를 내고 있기에, 현재 Vision에 사용되는 Transformer를 위주로 어떠한 연구들이 진행되고 있는지 보여주고 기초적인 Transformer에 대한 내용을 다룬다.

### Transformer란 무엇인가?

[트랜스포머(Transformer)는 2017년 구글이 발표한 논문인 "Attention is all you need"에서 나온 모델로 기존의 seq2seq의 구조인 인코더-디코더를 따르면서도, 논문의 이름처럼 어텐션(Attention)만으로 구현한 모델입니다. 이 모델은 RNN을 사용하지 않고, 인코더-디코더 구조를 설계하였음에도 번역 성능에서도 RNN보다 우수한 성능을 보여주었다.](https://wikidocs.net/31379)

Convolutional Neural Network(CNN), Recurrent Neural Network(RNN) 과 같이 이름 자체에 특정 기능들을 가지고 있다시피 Transformer의 경우 Encoder-Decoder를 이루고 있는 구조가 Narrow하게 dimension이 줄어들지 않고 변압기(Transformer)와 같이 생겼다고 붙혀진 이름이라고 생각한다.

![image](https://user-images.githubusercontent.com/37643248/161419184-ad2ec98f-a8ce-44e8-9306-0c3ecc313590.png)

[https://m.cafe.daum.net/funny-circuit/LfLC/2](https://m.cafe.daum.net/funny-circuit/LfLC/2)

![[http://machinelearningkorea.com/2019/07/09/트랜스포머-transformer와-어텐션-attention을-통해서-bert이해하기/](http://machinelearningkorea.com/2019/07/09/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B8-transformer%EC%99%80-%EC%96%B4%ED%85%90%EC%85%98-attention%EC%9D%84-%ED%86%B5%ED%95%B4%EC%84%9C-bert%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0/)](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F877650a2-2fef-46e6-8d53-36f8b15a6787%2FUntitled.png?table=block&id=d8fd4c1d-f9bb-41e8-bf25-34808fd95bbf&spaceId=481d9426-889b-4f3b-826f-aa895f410530&width=2000&userId=0d8a9c44-fd32-4ae6-b440-1997ded9b6bd&cache=v2)

[http://machinelearningkorea.com/2019/07/09/트랜스포머-transformer와-어텐션-attention을-통해서-bert이해하기/](http://machinelearningkorea.com/2019/07/09/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B8-transformer%EC%99%80-%EC%96%B4%ED%85%90%EC%85%98-attention%EC%9D%84-%ED%86%B5%ED%95%B4%EC%84%9C-bert%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0/)

### Transformer의 구성요소?

Transformer는 디테일한 과정을 제외한다면 크게 Seq-Seq 와 Attention 구조로 이루어져있다고 볼 수 있다. [https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

- Seq-Seq란
    - Sequence to Sequence란 두 개의 Recurrent Neural Network로 이루어진 모델로 input을 어떠한 feature representation으로 바꾸고 바꾸게 된 representation vector(context)를 decoder로 input으로 차례대로 넣어서 다시 word로 나타낼 수 있게 한다.
    - EOS와 SOS는 각각 문장의 종결, 시작으로 특정 index로 나타내는 것이고 Seq2Seq의 inference와 training stage 시의 동작 방식에 대한 차이점을 어느정도 이해하면 전체적은 흐름을 이해할 수 있다.
    
    ![image](https://user-images.githubusercontent.com/37643248/161419233-66723136-9b70-47a5-92f4-e558f1e99480.png)
    
- Attention이란
    - Attention은 영어말 그대로 어떤 것에 ‘집중’할 것이냐에 관한 것이다. Hidden input space에서 scalar value (0~1)를 multiply해서 hidden input space안에서 보지 않을 hidden value를 만들어 낼 수 있다.
    - [https://blog.floydhub.com/attention-mechanism/](https://blog.floydhub.com/attention-mechanism/)
    
    ![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fd8d0214e-a5fd-41f7-8c3e-21ddb21aa9d3%2FUntitled.png?table=block&id=0b7a2d3c-d278-497e-8f02-12e088c89889&spaceId=481d9426-889b-4f3b-826f-aa895f410530&width=2000&userId=0d8a9c44-fd32-4ae6-b440-1997ded9b6bd&cache=v2)
    
- Seq2Seq + Attention = Transformer?
    - 기본적인 구성요소는 seq2seq network과 attention algorithm을 가지고 있지만 이 의외에도 transformer만이 가지고 있는 여러가지 구성 요소들이 있다.
    - Positional encoding
        - input vector에 대해서 순서를 vector 형태로 집어 넣게 되면서 위치에 대한 중요성을 부여하는 개념
    - Point-wise Feed Forward Network
        - 네트워크 마지막 부분에 적용하는 두개의 dense 레이어
    - etc...

### 2022년 현재 Vision Transformer의 위치

![[https://paperswithcode.com/sota/image-classification-on-imagenet](https://paperswithcode.com/sota/image-classification-on-imagenet)](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F8244380e-0eff-4e9a-bee0-74ced1365328%2FUntitled.png?table=block&id=450331c0-068d-44db-9992-326a8d092868&spaceId=481d9426-889b-4f3b-826f-aa895f410530&width=2000&userId=0d8a9c44-fd32-4ae6-b440-1997ded9b6bd&cache=v2)

[https://paperswithcode.com/sota/image-classification-on-imagenet](https://paperswithcode.com/sota/image-classification-on-imagenet)

위는 ImageNet classfication task에 대해서 현재 state-of-the-art(SOTA) performance를 기록하고 있는 papers-with-code의 site이다. 1위의 경우 CoAtNet-7이라고 해서 [Neural Architecture Search(NAS)](https://en.wikipedia.org/wiki/Neural_architecture_search)라는 개념을 사용한 것이고 그것 바로 아래로 ViT-G/14로 Vision in Transformer에서 나오는 Vision Transformer를 사용하고 있다. 이외에도 다양한 [semantic segmentation](https://en.wikipedia.org/wiki/Image_segmentation), [depth estimation](https://en.wikipedia.org/wiki/Depth_perception) 등 다양한 분야에 대해서 좋은 성능을 보이고 있다.

- Various paper of ViT
    - AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE (Vision in Transformer) ([ICLR2021](https://openreview.net/forum?id=YicbFdNTTy))
        - Vision in Transformer같은 경우 위에서 Seq2Seq 개념에서 input에 각각의 단어가 들어가는 과정에서 image를 patch형태로 쪼개고 그 patch를 일정 pre-processing 개념으로 나눠서 transformer에 들어가게 된다.
        
        ```python
        # https://github.com/google-research/vision_transformer/blob/e8fae8228e877e45b560d6d70a11a59e061c106e/vit_jax/models.py#L255
        # Most basic concept of patch partioning
        
        n, h, w, c = x.shape
        
            # We can merge s2d+emb into a single conv; it's the same.
            x = nn.Conv(
                features=self.hidden_size,
                kernel_size=self.patches.size,
                strides=self.patches.size,
                padding='VALID',
                name='embedding')(
                    x)
        ```
        
        ![[https://theaisummer.com/vision-transformer/](https://theaisummer.com/vision-transformer/)](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F6d4e6020-25cd-44f2-93be-45e29a48cbfb%2FUntitled.png?table=block&id=8d290b33-3718-47e4-9c2c-2f603a924847&spaceId=481d9426-889b-4f3b-826f-aa895f410530&width=2000&userId=0d8a9c44-fd32-4ae6-b440-1997ded9b6bd&cache=v2)
        
        [https://theaisummer.com/vision-transformer/](https://theaisummer.com/vision-transformer/)
        
    - **BEiT: BERT Pre-Training of Image Transformers** ([Arxiv](https://arxiv.org/abs/2106.08254))
        - 기본적으로 Image Transformer를 학습시키기 위해서 많은 data들이 필요하기 때문에 그러한 점 또한 보완 하기 위해서 self-supervised learning중에 한 가지 방법과 비슷한 blockwise masking을 이용하여서 더 좋은 performance를 기록하게 하는 것을 소개 했다.
        
        ![image](https://user-images.githubusercontent.com/37643248/161419322-e6b337d9-c975-434f-9736-dd27f839e2c4.png)
        
        [https://arxiv.org/pdf/2106.08254.pdf](https://arxiv.org/pdf/2106.08254.pdf)
        
    - Swin Transformer : Hierarchical Vision Transformer using Shifted Windows ([ICCV 2021 Best paper](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html))
        - 기본적으로 Transformer는 mechanism 자체가 convolutional neural network와 비슷한 구조를 가지고 있지 않지만 coarse-to-fine 구조와 비슷하게 multi-level feature들을 고려하여서 최근 가지고 있는 model performance중에 가장 좋은 performance를 기록하고 있는 논문이다.
        - 2022/03/02 일 기준으로 조금 더 hyperparameter들의 tuning 밑 다양한 novelty를 통해서 Swin Transformer V2가 나오게 되었다.
        - Shifted window의 작동 방식의 경우 대표적으로 아래와 같이 transformerblock에서도 level 1에서 torch.roll을 통해서 진행한다음 level 2에서 다시 그대로 input을 집어 넣으면서 진행된다.
            
            ```python
            def window_partition(x, window_size):
                """
                Args:
                    x: (B, H, W, C)
                    window_size (int): window size
                Returns:
                    windows: (num_windows*B, window_size, window_size, C)
                """
                B, H, W, C = x.shape
                x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
                windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
                return windows
            
            def window_reverse(windows, window_size, H, W):
                """
                Args:
                    windows: (num_windows*B, window_size, window_size, C)
                    window_size (int): Window size
                    H (int): Height of image
                    W (int): Width of image
                Returns:
                    x: (B, H, W, C)
                """
                B = int(windows.shape[0] / (H * W / window_size / window_size))
                x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
                x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
                return x
            
            # https://github.com/microsoft/Swin-Transformer/blob/6ded2577413b68cbbd89f08391465788ed73030e/models/swin_transformer.py#L233
            # See the figure below (c)
            def forward(self, x):
                    H, W = self.input_resolution
                    B, L, C = x.shape
                    assert L == H * W, "input feature has wrong size"
            
            				# ResNet like shortcut
                    shortcut = x
            				# self.norm1 indicates Layernorm, https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
                    x = self.norm1(x)
                    x = x.view(B, H, W, C)
            
                    # cyclic shift
                    if self.shift_size > 0:
                        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                    else:
                        shifted_x = x
            
                    # partition windows purpose is to make patches of windows with the kernel normally used in PyTorch
                    **x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C**
                    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
            
                    # W-MSA/SW-MSA : SW-MSA means shifted window
                    attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
            
                    # merge windows
                    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
                    shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            
                    # reverse cyclic shift
                    if self.shift_size > 0:
                        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                    else:
                        x = shifted_x
                    x = x.view(B, H * W, C)
            
                    # FFN
                    x = shortcut + self.drop_path(x)
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
            
                    return x
            ```
            
        
        ![image](https://user-images.githubusercontent.com/37643248/161419371-cf2cfa0f-4a14-4495-ad06-865f7a1edef9.png)
        
        [https://github.com/microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
        
    - **MobileViT: Light-weight, General-purpose, and Mobile friendly Vision Transformer** ([Arxiv](https://arxiv.org/abs/2110.02178), ICLR 22)
        - 기존에 transformer의 network들이 mobilenetv2와 같은 lightweight network에 비해서 mobile device에서 optimization이 안되어 있는 경향을 보이는데 최대한 inference engine의 부하를 줄이기 위해서 여러가지 기법들을 사용한 논문들 중에 하나이다. 많은 novelty들이 들어있지는 않지만 속도적인 측면에서 우월성을 보인다.
        - 위 논문은 training 적인 부분 이외에도 coreml로 porting하는 과정까지 모두 공개가 되어 있어서 mobile implementation을 하기에 용이하다. ([Github](https://github.com/apple/ml-cvnets))
        - 일반적으로는 MobileNetV2의 작업들과 굉장히 비슷하고 PyTorch, Keras 모두 호환되는 공식코드가 있기에 실험적인 PoC에 사용해볼 수 있다.
        
        ![image](https://user-images.githubusercontent.com/37643248/161419393-d1270b2e-0bab-4bdf-b066-7580c89802ff.png)
        

### Conclusion

이 study의 목적이 applicable machine learning이기 때문에 특히나 그중에서도 on-device learning이라면 transformer 구조가 아직까지 FPGA 밑 Mobile device에서 다른 일반적인 neural network들에 비해 최적화가 되어 있지 않기 때문에 적합하다고 표현할 수는 없다. 그러나 GPU가 많이 발전함에 따라서 mobile에서 보여지는 machine learning result들이 server에서 통신이 되는 경우도 많이 발생하고 있는 상황이다. 그렇기에 Transformer가 다른 기존의 CDNN ()보다 좋은 성능을 보이기 때문에 (요즘에는 간간히 별로 차이가 없다라는 논문들도 보이긴 하지만) model performance를 위해서는 기존 base model에 대한 교체 밑 선택이 고려되어야 한다. 물론 Machine Learning Engineer Perspective 관점에서 CNN과 Transformer의 차이가 개념론적인 차이만 존재하고 원리는 그래도 같다라고 생각하는 사람으로써 연구적으로는 항상 Transformer를 사용해야하는지에 대한 의문점은 존재한다. CNN의 경우 Transformer와 달리 Attention(어떠한 feature에 대해서 집중을 할지)이 없는 구조로 형성되어 있다보니 현재 소위 말하는 XAI(eXplainable AI)의 영역에서 조금 힘든점이 존재한다. 그러나 [Revisiting ResNets: Improved Training and Scaling Strategies](https://arxiv.org/pdf/2103.07579.pdf)  이런 논문들과 같이 근본적으로 DL에 사용되는 모델 아키텍쳐에 대한 새로운 해석들과 학습 hyperparameter들에 대한 발전들이 계속해서 이루어지고 있기 때문에 결국 SOTA를 달성하는 모델들의 경우 현재 Machine Learning/Deep Learning Engineer들의 선호하는 Architecture에 따라서 혹은 유행에 따라서 계속 변화할 것이라고 생각한다.

### Appendix

[Transformer.pdf](https://github.com/applicable-ml/applicable-ml.github.io/files/8404090/Transformer.pdf)
