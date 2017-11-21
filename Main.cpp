#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <exception>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "boosting/AdaBoost.h"
#include "boosting/features/Data.h"
#include "boosting/features/HaarFeatures.h"
#include "boosting/utils/IntegralImage.h"
#include "boosting/utils/Utils.hpp"
#include "facedetector/FaceDetector.h"

using namespace std;
using namespace cv;

int main( int argc, char** argv ){

	string imagePath = "/Users/ArunSethupat/Documents/Sviluppo/Workspace/AdaBoost/dataset/";

	string positivePath = imagePath + "lfwcrop/faces/";
	string negativePath = imagePath + "backgrounds/";
	string validationPath = imagePath + "validation/";

	Mat test = imread(imagePath + "test/tammy.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	FaceDetector* detector = new FaceDetector("trainedData.txt", 1);
	detector->detect(test, true);

    return 0;
}
