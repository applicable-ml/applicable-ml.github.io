---
title: Custom TFLite Library for iOS Developer (English)
tags: ['TFLite', 'iOS', 'Universal Framework', 'Model Serving']
author: "Sangyeop Jeong"
---

> [Custom TFLite Library for iOS Developer (한국어)](https://applicable-ml.github.io/custom_tflite_library_for_ios_developer_kr/)

# Intro

: TFLite framework for iOS Developer is distributed in CocoaPod. So once you make the Podfile and specify the following, any developers can use the TFLite framework easily.

```ruby
# Swift #
use_frameworks!
pod ‘TensorFlowLiteSwift’

# Objective-C #
pod ‘TensorFlowLiteObjC’
```

 If you wanna make an Application using TFLite, You just need to use CocoaPod. But, How about developing and distributing libraries use TFLite? If you distribute your library via CocoaPod, you can simply specify its dependency with `TensorFlowLite(Swift/Objc)` in CocoaPod Spec. But. If you distribute your library through other channels, You need to build TensorFlowLite yourself.

 Also, TFLite was built to *.framework(instead of *.xcframework). Currently, *.framework isn’t preferable because Apple recommend to use *.xcframework. So, It is good to build to *.xcframework of TFLite yourself.

 This post is helpful for iOS Developer that build TFLite itself. This post is written as of TensorFlow 2.4, If you use another version, maybe need to modify some build script.

# Universal Library Build

: In TFLite Project, You can find build script of Universal Library for iOS(See this [Link](https://github.com/tensorflow/tensorflow/blob/v2.4.2/tensorflow/lite/tools/make/build_ios_universal_lib.sh)). Let’s assume you use x86_64 mac, run build script with specifying arguments. 

```bash
./build_ios_universal_lib.sh -a "x86_64 arm64"
```

after build completes, `libtensorflow-lite.a` will be generated.

# Add libtensorflow-lite.a in Xcode Project

## Add Header File

: Let’s import libtensorflow-lite.a in Xcode Project.

<img src="https://user-images.githubusercontent.com/17686601/161046218-305119b9-eaed-466b-940c-35a127c1697a.png" width="80%"/>

 Hmm... Can we build this project now? Sadly, there are some task yet. libtensorflow-lite.a is C/C++ library, so we have to import header files in Xcode Project.

 In this [Link](https://github.com/tensorflow/tensorflow/tree/v2.4.2/tensorflow/lite), you can find many *.c/cpp and header files of TFLite. We imported libtensorflow-lite.a previously, we just need to import header files. Unfortunately, some header is incompatible for iOS. Which header is incompatible? Seriously, we have to check one by one? Somewhere in the world, the great developer has already classified usable headers. Let’s visit this repository([Link](https://github.com/ValYouW/tflite-dist/releases/tag/v2.4.1)), then download and import header files.

## Specify Header Path

: importing header files, you have to specify Header Search Paths in Build Settings. Specify proper path according to your project.

<img src="https://user-images.githubusercontent.com/17686601/161046330-2a253a71-e877-478b-bade-449caa67a7ce.png" width="80%"/>

## Specify **Linker Flag**

: For using C/C++ library, specify Other Linker Flags. I specified `-lc++`.

<img src="https://user-images.githubusercontent.com/17686601/161046419-f53c8011-c27e-4518-894e-ab1ca2b87e09.png" width="80%"/>

# Conclusion

: Project setting is completed for building TFLite then importing `libtensorflow-lite.a`. Did you build successfully? The project setting gave me too much trouble. I hope that this post help to save your time.
