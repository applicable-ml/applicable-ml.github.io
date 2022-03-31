# Custom TFLite Library for iOS Developer

### **Intro**

: iOS 개발자가 Project에 TFLite를 사용할 수 있도록, TFLite는 CocoaPod으로도 배포되어 있습니다. Podfile을 작성하고, 아래와 같이 Pod을 명시하면 쉽게 TFLite를 Project에 추가할 수 있어요.

**Swift**

use_frameworks!

pod ‘TensorFlowLiteSwift’

**Objective-C**

pod ‘TensorFlowLiteObjC’

 TFLite를 이용하여 Application을 개발하고 AppStore에 배포하는 것이 목표라면 CocoaPod을 이용하는 것으로도 소기의 목적을 달성할 수 있을 겁니다. 

 그렇다면 TFLite를 이용하여 Library를 개발하고 배포하면 어떨까요? CocoaPod으로 배포할 경우, Library에 대한 CocoaPod Spec을 작성할 때 의존관계로 ‘TensorFlowLite...’를 명시할 수 있습니다. 하지만 CocoaPod이 아닌 다른 형태로 배포하는 경우에는 직접 TensorFlowLite를 Build 할 필요가 있어요.

 그리고 이 글을 작성하는 시점에선 CocoaPod으로 배포된 TFLite가 *.xcframework가 아닌 *.framework Format으로 배포되고 있습니다. 바람직하지 않은데, TFLite를 직접 Build 할 수 있으면 개선할 수 있는 부분입니다.

 이 글은 TFLite를 직접 Build 해서 사용하길 희망하는 iOS 개발자들에게 도움을 주기위해 작성했습니다. TensorFlow 2.4 Version을 기준으로 작성하여 그 이후 Version에서는 일부 Build Script가 변경되었을 수도 있지만, 조금만 수정하면 적용하는 데에 문제가 될 부분은 없을겁니다. 저의 시행착오가 여러분들의 문제 해결에 도움이 되길 희망하며, 방법을 공유하겠습니다.

### Universal Library Build

: TFLite Project에는 iOS용 Universal Library를 Build하는 Script가 있습니다. Script는 이 [Link](https://github.com/tensorflow/tensorflow/blob/v2.4.2/tensorflow/lite/tools/make/build_ios_universal_lib.sh)에서 보실 수 있습니다. x86_64 계열 Mac을 사용한다고 가정하고, 아래와 같이 argument를 명시해서 Script를 실행해 봅시다.

./build_ios_universal_lib.sh -a “x86_64 arm64”

Build 결과물로 libtensorflow-lite.a가 생성됩니다.

### **Add libtensorflow-lite.a in Xcode Project**

- **Header File 추가**

: libtensorflow-lite.a가 생성됬으니 Xcode Project에 추가해 봅시다.

<img src="https://user-images.githubusercontent.com/17686601/161046218-305119b9-eaed-466b-940c-35a127c1697a.png" width="80%"/>

 이제 Build를 할 수 있을까요? 아쉽게도 Build를 하기 위해 추가적인 작업이 필요합니다. libtensorflow-lite.a는 C/C++로 개발된 Library이고 수많은 *.c/cpp File들 외에도 수많은 Header File들도 있을 것입니다. 그렇습니다, Header File들을 Xcode Project에 추가해야 합니다.

 이 [Link](https://github.com/tensorflow/tensorflow/tree/v2.4.2/tensorflow/lite)에는 TFLite의 수많은 *.c/cpp 그리고 Header File들이 포함되어 있습니다. libtensorflow-lite.a는 이미 Xcode Project에 추가했으니 Header File만 추가하면 되는데요, 문제는 일부 Header들은 추가시 Build Error를 발생시킵니다. 그래서 지워야 하는데, 일일이 찾아서 지우는 것은 꽤 번거로운 작업이 되겠죠? 다행히도 쓸 수 있는 Header들만 분류한 Github Repository([Link](https://github.com/ValYouW/tflite-dist/releases/tag/v2.4.1))가 있습니다. 이 Header들만 Xcode Project에 추가하면 됩니다.

- **추가한 Header Path 명시**

: Header를 추가한 다음, Build Setting에서 Header Search Paths를 명시해주셔야 합니다. 각자 Project에 맞게 적절한 경로를 명시하시면 됩니다.

<img src="https://user-images.githubusercontent.com/17686601/161046330-2a253a71-e877-478b-bade-449caa67a7ce.png" width="80%"/>

- **Linker Flag 추가**

: libtensorflow-lite.a는 C/C++로 개발되었으므로, Linker Flag를 추가해주어야 합니다.

<img src="https://user-images.githubusercontent.com/17686601/161046419-f53c8011-c27e-4518-894e-ab1ca2b87e09.png" width="80%"/>

### **Conclusion**

: TFLite를 직접 Build하여 Xcode Project에 추가해서 사용하기 위해, 필요한 작업들이 끝났습니다. Build가 잘 되시나요? 저는 Build를 성공하는 데에 이런저런 시행착오가 있었는데요, 이 글이 여러 분들의 작업 시간을 절약할 수 있길 희망합니다.
