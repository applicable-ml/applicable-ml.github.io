---
title: iOS 개발자의 음성인식과 개체명인식 모델 사용기(feat. Core ML의 NLP, Speech 프레임워크)
tags: ['ios', 'ml', 'nlp', 'speech', 'coreml', 'ner', 'speech-recognition']
author: "Sungwook Baek"
---

이번 스터디 시즌2, 3 참여를 통해 Core ML로 현재 어디까지 할 수 있는지 이것저것 실험(?)을 해봤다.

이번 글을 통해 그 동안의 경험들을 정리해보았다.

## 내가 Core ML로 해보고 싶었던 것들

개인적으로 지출을 쉽게 기록할 수 있는 기능들을 만들고 싶었다.

예를 들면 음성 명령으로 정보를 입력. 

- 어디에서 무엇을 샀고 지출액이 총 얼마인지를 기록해줘.
    
    <img src="https://user-images.githubusercontent.com/37643248/164344080-8d413ca4-8020-4b2b-9073-242c0d000018.png" width="300px">
    
<img src="https://user-images.githubusercontent.com/37643248/164344156-1195c4ba-9d7e-43a4-923d-9006455c854f.png">

## 테스트

뭔가 모델을 직접 학습하기 전에 애플의 Core ML에서 기본적으로 제공해주는 기능들이 어느정도 수준까지 동작하는지 먼저 파악을 해보았다. 그리고 생각했던 것보다 결과가 잘 안나온 부분들은 별도로 모델 학습을 어떻게 해서 개선해야 될 지 고민해보았다.

### 첫 번째 테스트,  마이크로 목소리를 인식해서 텍스트로 변환해주는 Speech To Text.

[음성인식](https://en.wikipedia.org/wiki/Speech_recognition)(SST, Speech Recognition)은 [Speech Framework](https://developer.apple.com/documentation/speech)를 불러와서 사용할 수 있다. 애플에서 제공한 샘플 프로젝트를 간단하게 살펴보고 사용법을 이해해보자.

Apple의 공식 Speech 샘플 프로젝트

[Apple Developer Documentation](https://developer.apple.com/documentation/speech/recognizing_speech_in_live_audio)

```swift
public class ViewController: UIViewController, SFSpeechRecognizerDelegate {
		//한국어 인식을 위해 Locale 세팅을 ko-KR로 변경
    private let speechRecognizer = SFSpeechRecognizer(
				locale: Locale(identifier: "ko-KR")
		)!
    
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    
    private var recognitionTask: SFSpeechRecognitionTask?
    
    private let audioEngine = AVAudioEngine()
...
}
```

[SFSpeechRecognizerDelegate](https://developer.apple.com/documentation/speech/sfspeechrecognizer/1649893-delegate)는 음성인식 기능이 현재 디바이스에서 사용가능한 상태인지를 확인하는 용도로 사용한다. 예를 들어 디바이스의 인터넷 연결상태 혹은 아이폰 모델별로 이 기능을 사용할 수 있는지 여부등을 확인할 수 있다.

`isAvailable`

- 현재 음성인식 기능을 사용할 수 있는지 여부를 확인

`supportsOnDeviceRecognition`

- 현재 사용자의 아이폰 모델이 인터넷 연결 없이도 음성인식 기능을 사용할 수 있는지 여부를 확인

먼저 한국어 인식을 테스트하기 위해 `SFSpeechRecognizer`생성 시 언어 세팅 값을 한국어로 변경하자.

```swift
recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { result, error in
            var isFinal = false
            
            if let result = result {
                // 음성 인식된 결과를 출력
                self.textView.text = result.bestTranscription.formattedString
                isFinal = result.isFinal
                print("Text \(result.bestTranscription.formattedString)")
            }
            
            if error != nil || isFinal {
                // Stop recognizing speech if there is a problem.
                self.audioEngine.stop()
                inputNode.removeTap(onBus: 0)

                self.recognitionRequest = nil
                self.recognitionTask = nil

                self.recordButton.isEnabled = true
                self.recordButton.setTitle("Start Recording", for: [])
            }
        }
```

그리고 난 다음에 `recognitionTask`를 통해 결과를 화면에 출력할 수 있다.

### 음성인식 테스트 결과는?

<img src="https://user-images.githubusercontent.com/37643248/164344241-bafc1c40-0276-4837-a7bc-66300f53f41c.png">

한국어 인식이 꽤 잘 되는 것을 확인할 수 있었다. 특히 시간과 금액을 말했을 때 어떻게 변환되는지 과정을 살펴볼 수 있었는 데 정확하게 의도한대로 텍스트로 변환해주었다.

### Speech Framework 특징

- 온 디바이스에서 동작 (오프라인에서 사용 가능)
- 여러 나라의 언어 지원 (총 63개국 언어)

```swift
//지원하는 언어 목록 체크
let locales = SFSpeechRecognizer.supportedLocales()

//Output
[cs-CZ (fixed), da-DK (fixed), wuu-CN (fixed), de-DE (fixed), 
th-TH (fixed), en-ID (fixed), vi-VN (fixed), sk-SK (fixed), 
nl-BE (fixed), fr-CA (fixed), ca-ES (fixed), ro-RO (fixed), 
ms-MY (fixed), it-CH (fixed), uk-UA (fixed), fr-CH (fixed), 
tr-TR (fixed), en-SA (fixed), de-CH (fixed), hi-IN (fixed), 
zh-TW (fixed), en-ZA (fixed), nl-NL (fixed), es-CL (fixed), 
hu-HU (fixed), hr-HR (fixed), el-GR (fixed), ja-JP (fixed), 
en-AE (fixed), pt-PT (fixed), en-US (fixed), es-CO (fixed), 
hi-Latn (fixed), es-US (fixed), es-419 (fixed), yue-CN (fixed), 
en-CA (fixed), hi-IN-translit (fixed), en-IE (fixed), pt-BR (fixed), 
pl-PL (fixed), ru-RU (fixed), en-SG (fixed), de-AT (fixed), 
he-IL (fixed), en-GB (fixed), es-ES (fixed), sv-SE (fixed), 
id-ID (fixed), en-IN (fixed), it-IT (fixed), zh-HK (fixed), 
en-AU (fixed), ko-KR (fixed), fi-FI (fixed), zh-CN (fixed), 
fr-FR (fixed), es-MX (fixed), en-NZ (fixed), en-PH (fixed), 
ar-SA (fixed), fr-BE (fixed), nb-NO (fixed)]
```

Speech Framework를 이용하여 음성인식까지는 잘 되는 것을 확인했다. 즉 음성 인식 단계에서는 내가 별도로 추가적으로 인식율을 높이기 위한 작업이 필요없다는 것을 알았다. 아래 그림의 빨간색으로 표시한 부분까지는 확인한 셈이다.

<img src="https://user-images.githubusercontent.com/37643248/164344253-ec888262-ceee-42bd-b1f3-2b6db3b4ad92.png">

### Natural Language 프레임워크의 Word Tagger (NER, Named Entity Recogition) 테스트

1단계 테스트의 결과물은 텍스트이다. **“오늘 다섯시 정도에 종로카페에서 8200원을 지출했어”**가 결과물인데 이제 이 문장 안에서 `장소`,`시간`,`금액`을 분리해내는 일이 남았다.

<img src="https://user-images.githubusercontent.com/37643248/164344287-aaccfdfd-5465-45d4-8d35-11060e50003b.png">

<이미지 출처: [https://developer.apple.com/documentation/naturallanguage](https://developer.apple.com/documentation/naturallanguage)>

일단 한국어를 지원하는지 먼저 확인해보자.

```swift
let supportTags = NLTagger.availableTagSchemes(
    for: .word,
    language: .korean
).compactMap { $0.rawValue }

//Output
["Language", "Script", "TokenType"]
```

아쉽게도 한국어 관련된 정보분석은 아래와 같이 3가지 정도만 지원한다.

- `tokenType`
    - 입력 값이 단어인지 문장인지 등을 판단
- `language`
    - 입력 값이 한국어인지 영어인지 판단
- `script`
    - 입력값의 언어가 독일어면 `German` 중국어면 `Hant` 영어면 `Latin`이라고 알려준다. 한국어를 테스트 해보니 `Kore`라고 나온다. 이 부분이 조금 이상했다. `Korea`라고 나와야 되는거 아닌지?

```swift
let inputString = "한국어"
let tagger = NLTagger(
	tagSchemes: [.tokenType, .language, .script], 
	options: 0
)
tagger.string = inputString
let startIndex = inputString.startIndex

tagger.tag(at: startIndex, unit: .word, scheme: .tokenType) // Word
tagger.tag(at: startIndex, unit: .word, scheme: .language) // ko
tagger.tag(at: startIndex, unit: .word, scheme: .script) // Kore
```

일단 한국어는 장소, 가격, 동사인지 형용사인지 등등을 전혀 구분 못한다는 것을 확인했다. 추가적으로 `NLTagger`가 특정 언어만 인식하도록 세팅하고 싶으면 아래와 같이 `setLanguage`를 이용하면 된다.

```swift
let tagger = NLTagger(
    tagSchemes: [.nameTypeOrLexicalClass, .language]
)
let nsrange = NSRange(location: 0, length: inputText.count)
let range = Range(nsrange, in: inputText)!
tagger.setLanguage(.korean, range: range as Range)
```

마지막 테스트로 영어 문장은 어떻게 인식하는지 확인 해보자.

```swift
let inputText = "I ordered a coffee at starbucks. It was 7 bucks."
let tagger = NLTagger(
    tagSchemes: [.nameTypeOrLexicalClass, .language]
)
tagger.string = inputText
tagger.enumerateTags(
    in: inputText.startIndex..<inputText.endIndex,
    unit: .word,
    scheme: .nameTypeOrLexicalClass,
    options: [.omitPunctuation, .omitWhitespace]
) { tag, range in
    print("Tag: \(tag?.rawValue ?? "unknown") -> \(inputText[range])")
    return true
}

//Output
Tag: Pronoun -> I
Tag: Verb -> ordered
Tag: Determiner -> a
Tag: Noun -> coffee
Tag: Preposition -> at
Tag: Noun -> starbucks
Tag: Pronoun -> It
Tag: Verb -> was
Tag: Number -> 7
Tag: Noun -> bucks
```

영어는 그럭저럭 토큰별로 분류가 잘 되는 편이라고 느꼈으나 좀더 다양하게 분류하고 싶다면 별도의 모델을 학습시킬 필요가 있다. 예를 들어 문장에서 가격을 따로 추출하고 싶으면 모델 학습이 필요하다.

## 모델을 학습시켜 보자

일단 한국어 인식이 잘 안되는 것을 확인했으니 제대로 인식되도록 모델을 학습시켜보자.

### 학습 데이터 만들기

먼저 학습할 모델을 JSON 포멧으로 만들자. 목표는 한국어 중에서 시간, 금액, 동사를 인식하는 모델 만들기다.

`tokens`는 한 문장을 token별로 쪼개고 각 토큰이 의미하는 label을 지정하는 방식으로 학습시켰다.

```json
[
    {   
        "tokens": ["오늘", "다섯시", "정도에", "종로카페에서", "8200원을", "지출했어"],
        "labels": ["NONE", "TIME", "NONE", "PLACE", "PRICE", "VERB"]
    },
    {    
        "tokens": ["지금", "잠실에서", "8700원", "썼어"],
        "labels": ["NONE", "PLACE", "PRICE", "VERB"]
    },
    {    
        "tokens": ["어제", "1200원", "썼어"],
        "labels": ["NONE", "PRICE", "VERB"]
    },
    {    
        "tokens": ["방금", "식비로", "2000원", "사용함"],
        "labels": ["NONE", "NONE", "PRICE", "VERB"]
    },
    {
             
        "tokens": ["오늘", "낮에", "카페에서", "8200원을", "지출했어"],
        "labels": ["NONE", "NONE", "PLACE", "PRICE", "VERB"]
    },
    {    
        "tokens": ["저녁", "8시에", "스타벅스에서", "4300원", "썼어"],
        "labels": ["NONE", "TIME", "PLACE", "PRICE", "VERB"]
    },
    {    
        "tokens": ["아까", "낮에", "1200원", "썼어"],
        "labels": ["NONE", "NONE", "PRICE", "VERB"]
    },
    {    
        "tokens": ["방금", "식비로", "9000원", "사용함"],
        "labels": ["NONE", "NONE", "PRICE", "VERB"]
    }
]
```

위의 학습 방법은 띄어쓰기를 기준으로 한 어절별로 분류한 데이터이다. 

다른 방법으로는 형태소로 쪼개서 학습하는 방법이 있다. 아래는 형태소 단위로 쪼갠 예시이다. labels는 생략하였다. 이번 포스팅에서는 어절별로 분류해서 학습한 데이터 결과만을 다루었다. 형태소나 다른 방식으로 학습한 결과는 조금 더 실험을 하고 의미있는 결과가 나오면 업데이트 할 예정이다.

```json
[
    {
        "tokens": ["오늘", "다섯", "시", "에", "종로", "카페", "에서", "8200", "원", "을", "지출", "하", "ㅁ"],
        "labels": []
    },
    {    
        "tokens": ["지금", "잠실", "에서", "8700", "원", "을", "쓰", "었", "어"],
        "labels": []
    },
    {    
        "tokens": ["어제", "1200", "원", "쓰", "었", "음"],
        "labels": []
    },
    {    
        "tokens": ["방금", "식비", "로", "2000", "원", "사용", "하", "ㅁ"],
        "labels": []
    },
    {
             
        "tokens": ["오늘", "낮", "에", "카페", "에서", "8200", "원", "을", "지출", "하", "였", "어"],
        "labels": []
    },
    {    
        "tokens": ["저녁", "8", "시", "에", "스타벅스", "에서", "4300", "원", "을", "쓰", "었", "어"],
        "labels": []
    },
    {    
        "tokens": ["아까", "낮", "에", "1200", "원", "을", "사용", "하", "였", "어"],
        "labels": []
    },
    {    
        "tokens": ["방금", "식비", "로", "9000원", "사용", "하", "ㅁ"],
        "labels": []
    }
]
```

### 모델 생성하기

[Create ML](https://developer.apple.com/machine-learning/create-ml/) 프로그램을 실행. (Xcode → Open Developer Tool)

<img src="https://user-images.githubusercontent.com/37643248/164344335-0bd13045-7566-4648-bd99-07ded15d25bb.png"> | <img src="https://user-images.githubusercontent.com/37643248/164344349-585f4f3f-a05a-4bb0-813d-1f8eb4e27fdd.png">
-- | --
 | 

Text → Word Tagging을 선택

<img src="https://user-images.githubusercontent.com/37643248/164344426-e8ebf7cd-707f-4491-b9d9-4b5218dda766.png">

학습 데이터인 ko_tagger.json 파일을 선택하고 언어는 한국어로 설정했다. 그리고 Train 버튼을 누르면 모델이 생성된다.

<img src="https://user-images.githubusercontent.com/37643248/164344442-8113d11a-9a56-4b58-ab77-dac258370834.png">

Preview 메뉴를 선택하면 학습된 모델이 제대로 인식을 하는지 테스트를 할 수 있다. 일단 시간과 금액을 인식하는 것을 확인할 수 있었다. 모델을 다운로드 받으려면 Output으로 가서 Core ML 모델(`.mlmodel`)을 다운로드 받으면 된다.

<img src="https://user-images.githubusercontent.com/37643248/164344472-ff30790d-1d87-47de-991d-9a3d1073c817.png">

MLModel을 다운로드 받았으면 Playground 앱을 실행해서 macOS를 선택하고 모델 파일은 Resource 폴더로 드래그 앤 드롭해서 옮겨주자. 그리고 아래와 같은 코드로 내가 만든 모델의 인식 결과를 확인해볼 수 있다.

```swift
import NaturalLanguage
import CoreML

let text = "오늘 다섯시 정도에 종로카페에서 8200원을 지출했어"

do {
    let mlModel = try KoTagger(configuration: MLModelConfiguration()).model

    let customModel = try NLModel(mlModel: mlModel)
    let customTagScheme = NLTagScheme("KoTag")
    
    let tagger = NLTagger(tagSchemes: [.nameType, customTagScheme])
    tagger.string = text
    tagger.setModels([customModel], forTagScheme: customTagScheme)
    
    tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word,
                         scheme: customTagScheme, options: .omitWhitespace) { tag, tokenRange  in
        if let tag = tag {
            print("Tag: \(tag.rawValue) -> \(text[tokenRange])")
        }
        return true
    }
} catch {
    print(error)
}
```

완벽하지는 않지만 한국어를 인식하는 word tagger 모델을 직접 생성해서 어느정도 동작하는 것을 확인할 수 있었다. 학습모델만 조금 더 제대로 만들어서 MLModel을 생성한다면 분명 훨씬 더 좋은 인식결과가 나올 것 같다는 생각이 들었다.

<img src="https://user-images.githubusercontent.com/37643248/164344472-ff30790d-1d87-47de-991d-9a3d1073c817.png">

## 마치며

음성인식과 문장인식에 관심이 많았는데 Core ML로 정말 쉽고 간단하게 구현할 수 있었다. 특히 모델을 만드는 [Create ML](https://developer.apple.com/machine-learning/create-ml/) 툴이 정말 간편했고 내가 만든 모델을 코드로 직접 테스트 하기전에 Preview에서 확인할 수 있다는 점도 편리했다. 

Natural Language의 경우 아직 한국어는 Word Tagger 기능이 다른 언어에 비해 제한적인데 Word Tagger 모델을 직접 만들어서 해결할 수 있기 때문에 큰 문제는 없었다.

마지막으로 오프라인에서 동작하면서도 앱 용량에 크게 영향을 주지 않는 다는 점이 가장 큰 장점으로 느껴졌다.

## 관련 자료

- [https://developer.apple.com/documentation/naturallanguage/nltagscheme](https://developer.apple.com/documentation/naturallanguage/nltagscheme)
- [https://developer.apple.com/documentation/createml/creating_a_word_tagger_model](https://developer.apple.com/documentation/createml/creating_a_word_tagger_model)
- [https://en.wikipedia.org/wiki/Named-entity_recognition](https://en.wikipedia.org/wiki/Named-entity_recognition)
- [https://wikidocs.net/30682](https://wikidocs.net/30682)
- [https://paperswithcode.com/task/named-entity-recognition-ner](https://paperswithcode.com/task/named-entity-recognition-ner)
- [https://jins-sw.tistory.com/6](https://jins-sw.tistory.com/6)