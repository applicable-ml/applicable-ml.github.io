---
title: Core ML 기초
tags: ['ios', 'ml', 'coreml']
author: "Seonghun Kim"
---

## Intro
  현재 우리가 사용하는 대부분의 앱에서는 분야를 막론하고 머신러닝이 사용되고 있습니다. 오히려 머신러닝을 사용하지 않는 서비스를 찾기 힘들 정도죠. 앱에 머신러닝을 적용하는 방법은 여러 가지가 있습니다. 서버에 데이터를 전달하여 서버에서 머신러닝을 활용하여 원하는 결과를 가져오기도 하고 스마트폰 내에서 모델을 가지고 원하는 결과를 얻을 수도 있습니다.

  이 글에서는 iOS 앱에서 On-device 머신러닝을 적용하는 방법의 하나인 Core ML에 대해 살펴보려고 합니다. Core ML의 기본적인 사용법부터 원격으로 모델을 전달받는 방법 등 처음 Core ML을 앱에 적용할 때 필요한 내용을 다루고 있습니다. Core ML로 앱에 새로운 기능을 제공하고 싶으신 분들에게 도움이 되길 바랍니다.

## Core ML은 무엇인가?
  [Core ML](https://developer.apple.com/kr/machine-learning/core-ml/)은 머신러닝(Machine Learning)을 Apple platform에서 쉽게 사용할 수 있도록 돕는 프레임워크 입니다. 단지 Apple이 만든 Create ML로 학습한 모델 뿐만 아니라 다양한 도구로 학습한 머신러닝 모델들도 사용이 가능하도록 Core ML 모델로 변환하는 [converter](https://coremltools.readme.io/docs)도 함께 제공하고 있습니다.

  Core ML은 [Accelerate](https://developer.apple.com/documentation/accelerate)와 [Metal](https://developer.apple.com/documentation/metal/) framework 기반으로 제작되었으며, On-device 환경에 최적화하여 memory와 전력 사용량을 최소화 하였다고 합니다. 또한, On-device 환경에서 동작하기 때문에 Network 사용이 필요하지 않으며, 사용자 보안과 반응 속도에 도움이 된다고 Apple이 설명하고 있습니다. 

  Apple은 Core ML과 더불어 Core ML기반의 4개의 머신러닝 framework를 제공하고 있습니다. 이 framework를 활용하면 학습된 모델이 없더라도 앱에서 머신러닝을 활용한 기능을 구현할 수 있습니다.

  이 글에서는 모델의 학습 및 Core ML이 아닌 다른 모델을 Core ML 모델로 변환하는 내용은 포함하지 않습니다. Core ML 모델을 프로젝트에 어떻게 적용할 수 있을지 기본적인 사용법을 다룰 예정입니다.

## Core ML 기본적인 활용법
  학습된 모델이 없이 Core ML을 사용하려면 크게 두가지 방법이 있습니다.

  첫째는 [Apple이 제공하는 Core ML 모델](https://developer.apple.com/kr/machine-learning/models/)들을 활용하는 방법입니다. 풍족하진 않지만 꽤 많은 모델을 지원하며  처음 Core ML을 프로젝트에 도입하기엔 충분합니다. 

  둘째는 Apple이 Core ML 기반으로 구현한 Framework를 활용할 수 있습니다. 이미지와 비디오 관련 기능을 제공하는 [Vision](https://developer.apple.com/documentation/vision), 자연어 처리를 할 수 있는 [Natural Language](https://developer.apple.com/documentation/naturallanguage), 라이브나 녹화된 사람의 오디오를 인식할 수 있는 [Speech](https://developer.apple.com/documentation/speech), 어떤 소리를 식별할 수 있는 [Sound Analysis](https://developer.apple.com/documentation/soundanalysis) 등이 있습니다. 이 Framework들을 활용할 경우 별도의 모델을 학습시킬 필요가 없음은 물론이고 이미 구현된 다양한 딥러닝/머신러닝 기반의 기능들을 빠르게 프로젝트에 도입할 수 있습니다. 

  우선 Apple이 제공하는 Core ML 모델을 활용하여 어떻게 프로젝트에 적용할 수 있는지 부터 살펴보겠습니다. 이 예시에서는 Apple이 제공하는 [MobileNet](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet) 모델을 프로젝트에 추가해 보겠습니다. MobileNet은 이미지 분류 모델 중 하나 입니다.

  Core ML 모델은 `.mlmodel`를 확장자로 가집니다. 그리고 모델을 프로젝트에 추가하게 되면 자동으로 파일명의 class가 자동생성 됩니다.

<img src="https://user-images.githubusercontent.com/40361234/164222629-e07cd4db-4726-4029-8dba-0a5602ad0648.png" width=600px>

  자동생성된 클래스는 image와 함게 생성된 Input을 전달하여 Output의 형태로 추론 결과를 가져올 수 있습니다. 여기서 label은 식별할 물체를 나타내고 confidence는 확률값을 나타냅니다.
~~~ swift
let defaultConfig = MLModelConfiguration()
let imageClassifier = try! MobileNet(configuration: defaultConfig)
    
let input = try! MobileNetInput(imageWith: image)
let output = try! imageClassifier.prediction(input: input)
let predictions = output.classLabelProbs
    .sorted(by: { $0.value > $1.value })
    .first!
    
return (
    label: predictions.key,
    confidence: predictions.value
)
~~~

  앞서 예시를 보면 총 3가지 클래스가 자동생성 되는 것을 알 수 있습니다. 이 예제에서는 `MobileNet`, `MobileNetInput`, `MobileNetOutput` 입니다. 클래스의 이름은 모델의 파일명에 따라 달라집니다. 각 클래스에 대해 좀 더 살펴보겠습니다.

### Class for model loading and prediction

  첫번째는 모델을 불러오고 추론을 수행해주는 객체 입니다. 위 예제에서 `MobileNet` 입니다. 구현체를 간략하게 보면 다음과 같습니다.

```swift
class MobileNet {
    let model: MLModel

    class var urlOfModelInThisBundle : URL {
        let bundle = Bundle(for: self)
        return bundle.url(forResource: "MobileNet", withExtension:"mlmodelc")!
    }

    init(model: MLModel) {
        self.model = model
    }

    func prediction(input: MobileNetInput, options: MLPredictionOptions) throws -> MobileNetOutput {
        let outFeatures = try model.prediction(from: input, options:options)
        return MobileNetOutput(features: outFeatures)
    }

    func predictions(inputs: [MobileNetInput], options: MLPredictionOptions = MLPredictionOptions()) throws -> [MobileNetOutput] {
        let batchIn = MLArrayBatchProvider(array: inputs)
        let batchOut = try model.predictions(from: batchIn, options: options)
        var results : [MobileNetOutput] = []
        results.reserveCapacity(inputs.count)
        for i in 0..<batchOut.count {
            let outProvider = batchOut.features(at: i)
            let result =  MobileNetOutput(features: outProvider)
            results.append(result)
        }
        return results
    }
}
```

> `urlOfModelInThisBundle`: 프로젝트 내에 Model의 위치를 나타내는 URL
>
> `prediction(input:, options:)`: `Input`과 `MLPredictionOptions`를 받아 결과값의 Output을 반환하는 함수
>

  코드는 간단합니다. `MLModel`을 받아 `model` 변수에 저장하고 `prediction` 함수가 호출되면 추론후 Output 형태(예시에서는 `MobileNetOutput`)로 반환합니다. 

  이 코드만 보면 앞서 살펴본 사용예시와 다른 부분이 있습니다. 위 예시는 기본적인 부분만 가져온 것으로 다양한 `convenience init`이 함께 자동생성 됩니다. 

```swift
@available(macOS 10.14, iOS 12.0, tvOS 12.0, watchOS 5.0, *)
convenience init(configuration: MLModelConfiguration) throws {
    try self.init(contentsOf: type(of:self).urlOfModelInThisBundle, configuration: configuration)
}

convenience init(contentsOf modelURL: URL) throws {
    try self.init(model: MLModel(contentsOf: modelURL))
}
```

  `prediction` 함수도 마찬가지로 다양한 인자를 전달하여 추론할 수 있는 함수들이 함께 생성됩니다. 이러한 코드를 모두 담기엔 글이 불필요하게 길어져 간략한 부분만 표시하도록 하겠습니다.

### Model Prediction Input Type

 다음으로 모델 추론의 입력값으로 사용하는 객체 입니다. 위 예제에서는 `MobileNetInput` 입니다. 구현체를 간략하게 보면 다음과 같습니다.

```swift
class MobileNetInput : MLFeatureProvider {

    var image: CVPixelBuffer

    var featureNames: Set<String> {
        get {
            return ["image"]
        }
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        if (featureName == "image") {
            return MLFeatureValue(pixelBuffer: image)
        }
        return nil
    }
    
    init(image: CVPixelBuffer) {
        self.image = image
    }
}
```

> 기본 생성자는 `CVPixelBuffer`를 전달받지만 `CGImage`와 이미지 `URL`을 전달받아 생성하는 `convenience init`도 함께 자동 생성되어 있습니다.
>

  `Input`은 `MLFeatureProvider`를 상속받고 있습니다. `featureNames`와 `featureValue`는 해당 protocol에서 요구되고 있습니다.

```swift
/*
 * Protocol for accessing a feature value for a feature name
 */

@available(iOS 11.0, *)
public protocol MLFeatureProvider {

    var featureNames: Set<String> { get }
    
    /// Returns nil if the provided featureName is not in the set of featureNames
    func featureValue(for featureName: String) -> MLFeatureValue?
}
```

  `MLFeatureProvider`은 protocol로 모델의 입력과 출력을 나타냅니다.

  `MLFeatureProvider`에서 요구하는 `featureNames`은 입력 타입을 나타내고 `featureValue`는 실제 입력값을 제공합니다. 앞서 `MobileNet` class에서 보셨듯이 `MLModel`의 `prediction` 함수는 `MLFeatureProvider`를 상속받은 임의의 타입을 입력값으로 요구합니다. 그래서 `Input`도 `MLFeatureProvider`를 상속받아 구현된 것을 확인하실 수 있습니다.

### Model Prediction Output Type

  다음은 모델 추론의 결과값으로 사용하는 객체 입니다. 위 예제에서는 `MobileNetOutput` 입니다. 구현체는 다음과 같습니다.

```swift
class MobileNetOutput : MLFeatureProvider {

    private let provider : MLFeatureProvider

    lazy var classLabelProbs: [String : Double] = {
        [unowned self] in return self.provider.featureValue(for: "classLabelProbs")!.dictionaryValue as! [String : Double]
    }()

    lazy var classLabel: String = {
        [unowned self] in return self.provider.featureValue(for: "classLabel")!.stringValue
    }()

    var featureNames: Set<String> {
        return self.provider.featureNames
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        return self.provider.featureValue(for: featureName)
    }

    init(classLabelProbs: [String : Double], classLabel: String) {
        self.provider = try! MLDictionaryFeatureProvider(dictionary: ["classLabelProbs" : MLFeatureValue(dictionary: classLabelProbs as [AnyHashable : NSNumber]), "classLabel" : MLFeatureValue(string: classLabel)])
    }

    init(features: MLFeatureProvider) {
        self.provider = features
    }
}
```

> `classLabelProbs`: 추론값과 확률을 나타내는 `Dictionary`
>
> `classLabel`: 가장 높은 확률의 결과값
>

  `Output`도 마찬가지로 `MLFeatureProvider`를 상속받았습니다. 그리고 추론한 결과값을 가져오기 쉽도록 `classLabel`와 `classLabelProbs`를 제공하고 있습니다. 앞서 살펴본 `MobileNet` 코드를 보면 추론한 결과값으로 `Output`을 생성하는 걸 확인하실 수 있습니다. 이 결과값도 `MLFeatureProvider`를 상속받고 있으며 `Output`은 `MLFeatureProvider`에서 결과값을 쉽게 가져올 수 있도록 돕는 Wrapper 객체라고 할 수 있습니다.

  여기까지 간단한 Core ML 사용법과 자동생성되는 코드들을 함께 살펴보았습니다. 일반적으론 프로젝트에 모델을 추가하고 자동생성된 코드를 그대로 사용하면 됩니다. 

  하지만 여기서 한가지 의문이 생깁니다. 꼭 자동생성된 코드를 사용해야 할까요? 직접 구현한다면 모델을 프로젝트에 포함하지 않고도 Core ML을 사용할 수 있을까요?

  결론은 가능합니다. 앞서 살펴본 자동생성된 코드를 잘 살펴보시면 `MLModel`만 생성하여 주입해 주면 Core ML 기능을 사용할 수 있습니다. 그 방법을 좀 더 살펴보겠습니다.

## 원격으로 Core ML 모델을 변경하는 법

  앞서 `MobileNet` 클래스의 `urlOfModelInThisBundle`를 보면 어느 위치에 어떤 확장자로 저장되어 있는지 힌트를 얻을 수 있습니다. 프로젝트에 포함된 모델은 동일한 파일이름에 `.mlmodelc` 확장자로 저장되어 있습니다.

  여기서 Core ML 모델인 `.mlmodel`을 `.mlmodelc`로 변환해야 함을 알 수 있습니다. 자세한 방법은 Apple이 제공하는 [Downloading and Compiling a Model on the User’s Device](https://developer.apple.com/documentation/coreml/downloading_and_compiling_a_model_on_the_user_s_device) 문서에서 확인할 수 있습니다. 이 글에서는 간단하게 알아보도록 하겠습니다.

  우선 `.mlmodel`와 `.mlmodelc`의 차이를 알아보겠습니다. `.mlmodel`는 사용하기 전에 컴파일해야 하고 컴파일된 파일이 `.mlmodelc`입니다. 프로젝트에 포함된 경우 런타임에 컴파일을 하지 않고 미리 컴파일해서 포함됩니다. 하지만 만약 `.mlmodel`을 서버에서 다운받을 경우는 어떻게 해야 할까요? 이 경우 런타임에에 컴파일을 해야 합니다.

### Compile on runtime

  `.mlmodel`을 컴파일할 수 있는 API를 제공합니다. `MLModel`의 class method로 선언되어 있으며 컴파일된 파일 위치를 나타내는 `URL`을 반환합니다.

```swift
let compiledModelURL = try? MLModel.compileModel(at: url)
```

  그리고 컴파일된 모델의 `URL`로 `MLModel` 객체를 생성할 수 있습니다.

```swift
let imageClassifierModel = try? MLModel(contentsOf: compiledModelURL)
```

  `MLModel`을 생성하면 앞서 살펴본 자동생성된 클래스에서와 동일한 방법으로 사용하면 됩니다.

```swift
class MobileNet {
    let model: MLModel

    class var urlOfModelInThisBundle : URL {
        let bundle = Bundle(for: self)
        return bundle.url(forResource: "MobileNet", withExtension:"mlmodelc")!
    }

		// MLModel 객체로 초기화 한다.
    init(model: MLModel) {
        self.model = model
    }

    func prediction(input: MobileNetInput, options: MLPredictionOptions) throws -> MobileNetOutput {
        let outFeatures = try model.prediction(from: input, options:options)
        return MobileNetOutput(features: outFeatures)
    }

    func predictions(inputs: [MobileNetInput], options: MLPredictionOptions = MLPredictionOptions()) throws -> [MobileNetOutput] {
        let batchIn = MLArrayBatchProvider(array: inputs)
        let batchOut = try model.predictions(from: batchIn, options: options)
        var results : [MobileNetOutput] = []
        results.reserveCapacity(inputs.count)
        for i in 0..<batchOut.count {
            let outProvider = batchOut.features(at: i)
            let result =  MobileNetOutput(features: outProvider)
            results.append(result)
        }
        return results
    }
}
```

  한번 컴파일된 모델은 저장해 두고 매번 다시 사용할 수 있습니다. 이때 한가지 주의할 점은 처음 컴파일된 파일이 저장되는 위치는 Temporary directory입니다. Temporary directory에 보관된 파일들은 앱이 종료되면 삭제될 수 있습니다. 그렇기 때문에 재사용을 원한다면 컴파일된 모델을 다른 위치로 미리 옮겨둬야 합니다. Temporary directory에 대한 자세한 내용은 [File System Programming Guide](https://developer.apple.com/library/archive/documentation/FileManagement/Conceptual/FileSystemProgrammingGuide/FileSystemOverview/FileSystemOverview.html#//apple_ref/doc/uid/TP40010672-CH2-SW12)를 참고해 주세요.

  지금까지 우리는 Core ML을 사용하는 기본적인 방법을 알아봤습니다. Core ML 모델로 `MLModel` 객체를 생성하는 방법과 `MLModel`로 추론하는 방법을 자동생성된 코드를 통해 살펴봤습니다. 

  마지막으로 한가지 더 의문이 있을 수 있습니다. 만약 모델을 다운로드 받을 경우는 자동생성된 코드를 직접 구현하거나 비슷한 방법으로만 사용할 수 있을까요? Apple이 제공하는 Core ML 기반의 framework들에 내가 학습시킨 모델을 사용할 수 없을까요?

  이것도 가능합니다. 내장된 모델을 사용할 수 있지만 우리가 학습시킨 모델로 `MLModel`을 생성하여 주입할 수도 있습니다.

## Core ML 기반 프레임워크

  앞서 살펴봤듯이 Apple은 다양한 Core ML 기반 framework를 제공하고 있습니다. 또한 Core ML 모델을 주입할 수 있는 API를 함께 제공하고 있습니다. 이 글에서는 Vision framework의 이미지 분류 예시를 살펴보겠습니다.

<img src="https://docs-assets.developer.apple.com/published/8905e2c376/rendered2x-1636573848.png" width=400px>

이미지 출처: [https://developer.apple.com/documentation/coreml](https://developer.apple.com/documentation/coreml)

  우선 모델을 주입하지 않고 사용하는 방법을 살펴보겠습니다. 이 글의 목적이 Vision framework에 대한 것이 아닌 만큼 간단하게만 살펴볼 예정입니다.

```swift
let request = VNClassifyImageRequest(
    completionHandler: { (request: VNRequest, error: Error?) -> Void in
        let result = (request.results as! [VNClassificationObservation])
            .sorted(by: { $0.confidence > $1.confidence })
            .first!
            
        completion(result.identifier, result.confidence)
    }
)
    
let handler = VNImageRequestHandler(cgImage: image)
try! handler.perform([request])
```

> [`VNClassifyImageRequest`](https://developer.apple.com/documentation/vision/vnclassifyimagerequest): `VNImageBasedRequest`를 상속받은 객체로 이미지 식별 요청
>
> [`VNImageRequestHandler`](https://developer.apple.com/documentation/vision/vnimagerequesthandler): 이미지를 하나 또는 여러개의 분석 요청으로 처리하는 객체
>

  Vision framework를 사용할 땐 크게 두가지가 필요합니다. Request와 Handler입니다. Request는  `VNRequest`를 상속받은 객체들이며 위에서 언급된 `VNImageBasedRequest`도 `VNRequest`를 상속받았습니다. 

  Handler는 분석할 이미지와 `option`을 인자로 받아 초기화합니다. 그리고 Request 배열을 인자로 받아 `perform` 함수를 실행하면 이미지를 분석하고 결과값을 `request`의 `results` 변수에 담게 됩니다. 그 결과는 Request의 `completionHandler`로 전달받을 수 있습니다.

  모델을 주입할 경우 Request를 변경해 줘야 합니다. 먼저 코드를 살펴보겠습니다. 

```swift
let model = try! MLModel(contentsOf: compiledModelURL)
let visionModel = try! VNCoreMLModel(for: model)
    
let request = VNCoreMLRequest(
    model: visionModel,
    completionHandler: { (request: VNRequest, error: Error?) -> Void in
        let result = (request.results as! [VNClassificationObservation])
            .sorted(by: { $0.confidence > $1.confidence })
            .first!
            
        completion(result.identifier, result.confidence)
    }
)
    
let handler = VNImageRequestHandler(cgImage: image)
try! handler.perform([request])
```

> [`VNCoreMLModel`](https://developer.apple.com/documentation/vision/vncoremlmodel): `MLModel`을 Vision framwork에서 사용하기 위해 캡슐화한 객체, 적합하지 않은 모델로  초기화를 시도할 경우 에러를 반환한다.
>
> [`VNCoreMLRequest`](https://developer.apple.com/documentation/vision/vncoremlrequest): Core ML 모델을 사용하여 이미지 분석을 요청하는 객체
>

  앞서 모델을 주입하지 않는 경우와 다른 점은 `VNClassifyImageRequest`대신 `VNCoreMLRequest`를 전달하고 있습니다. `VNCoreMLRequest`이 생성 될 땐 `MLModel`을 캡슐화한 `VNCoreMLModel`을 인자로 받습니다. 그 외에는 사용법이 동일합니다. 만약 `VNCoreMLModel`를 생성 할 때 자연어 처리등 Vision framework에서 처리할 수 없는 `MLModel`을 입력받을 경우 error를 반환합니다.

  여기까지 Vision framework에 모델을 주입하는 방법을 알아 봤습니다. 여기서 한가지 주의할 점은 framework에 따라 모델을 주입해서 사용하는 방법이 다릅니다. 예를 들어 Natural Language framework의 경우 `MLModel`로 [`NLModel`](https://developer.apple.com/documentation/naturallanguage/nlmodel)을 생성하여 사용하고 있습니다. 그렇기 때문에 각 framework에서의 사용법은 Apple 문서를 참고 바랍니다.

* * *

  지금까지 Core ML의 기본적인 사용법을 알아봤습니다. 이 글에서는 이미지 분류 모델 만을 예시로 사용하였으나 실제론 다양한 머신러닝 모델이 존재합니다. 그 모델에 따라 입력값도 다르고 출력값이 다르기 때문에 이 글의 예시를 그대로 사용할 순 없습니다. 하지만 처음 Core ML을 접하고 공부하실 때 이 글이 도움이 되시길 바랍니다.

* * *

### 참고

- [https://developer.apple.com/kr/machine-learning/core-ml/](https://developer.apple.com/kr/machine-learning/core-ml/)
- [https://developer.apple.com/documentation/coreml](https://developer.apple.com/documentation/coreml)
- [https://developer.apple.com/videos/play/wwdc2017/703/](https://developer.apple.com/videos/play/wwdc2017/703/)
- [https://coremltools.readme.io/docs](https://coremltools.readme.io/docs)
- [https://developer.apple.com/documentation/metal/](https://developer.apple.com/documentation/metal/)
- [https://developer.apple.com/documentation/accelerate](https://developer.apple.com/documentation/accelerate)
- [https://developer.apple.com/kr/machine-learning/models/](https://developer.apple.com/kr/machine-learning/models/)
- [https://developer.apple.com/documentation/coreml/downloading_and_compiling_a_model_on_the_user_s_device](https://developer.apple.com/documentation/coreml/downloading_and_compiling_a_model_on_the_user_s_device)
- [https://developer.apple.com/library/archive/documentation/FileManagement/Conceptual/FileSystemProgrammingGuide/Introduction/Introduction.html#//apple_ref/doc/uid/TP40010672](https://developer.apple.com/library/archive/documentation/FileManagement/Conceptual/FileSystemProgrammingGuide/Introduction/Introduction.html#//apple_ref/doc/uid/TP40010672)
- [https://developer.apple.com/documentation/vision/classifying_images_for_categorization_and_search](https://developer.apple.com/documentation/vision/classifying_images_for_categorization_and_search)
- [https://developer.apple.com/documentation/vision/vnclassifyimagerequest](https://developer.apple.com/documentation/vision/vnclassifyimagerequest)
- [https://developer.apple.com/documentation/vision/vnimagerequesthandler](https://developer.apple.com/documentation/vision/vnimagerequesthandler)
- [https://developer.apple.com/documentation/vision/vncoremlmodel](https://developer.apple.com/documentation/vision/vncoremlmodel)
- [https://developer.apple.com/documentation/vision/vncoremlrequest](https://developer.apple.com/documentation/vision/vncoremlrequest)
- [https://developer.apple.com/documentation/naturallanguage/nlmodel](https://developer.apple.com/documentation/naturallanguage/nlmodel)
- [https://developer.apple.com/documentation/soundanalysis/classifying_sounds_in_an_audio_file](https://developer.apple.com/documentation/soundanalysis/classifying_sounds_in_an_audio_file)