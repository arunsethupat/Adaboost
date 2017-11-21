#ifndef BOOSTING_UTILS_INTEGRALIMAGE_H_
#define BOOSTING_UTILS_INTEGRALIMAGE_H_
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
class IntegralImage {
public:
	static Mat computeIntegralImage(Mat img);
	static Mat computeIntegralSquaredImage(Mat img, float mean);
	static float computeArea(Mat intImg, Rect r);
};
#endif
