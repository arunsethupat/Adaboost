cmake_minimum_required(VERSION 3.8)
project(AdaBoost_master)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        boosting/classifiers/CascadeClassifier.cpp
        boosting/classifiers/CascadeClassifier.h
        boosting/classifiers/Stage.cpp
        boosting/classifiers/Stage.h
        boosting/classifiers/StrongClassifier.cpp
        boosting/classifiers/StrongClassifier.h
        boosting/classifiers/WeakClassifier.cpp
        boosting/classifiers/WeakClassifier.h
        boosting/features/Data.cpp
        boosting/features/Data.h
        boosting/features/HaarFeatures.cpp
        boosting/features/HaarFeatures.h
        boosting/utils/Face.cpp
        boosting/utils/Face.h
        boosting/utils/IntegralImage.cpp
        boosting/utils/IntegralImage.h
        boosting/utils/Utils.hpp
        boosting/AdaBoost.cpp
        boosting/AdaBoost.h
        boosting/ViolaJones.cpp
        boosting/ViolaJones.h
        facedetector/FaceDetector.cpp
        facedetector/FaceDetector.h
        Main.cpp)

add_executable(AdaBoost_master ${SOURCE_FILES})