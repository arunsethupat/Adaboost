#ifndef BOOSTING_CLASSIFIERS_WEAKCLASSIFIER_H_
#define BOOSTING_CLASSIFIERS_WEAKCLASSIFIER_H_

#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include "../features/Data.h"
#include "../utils/Utils.hpp"

using namespace std;
using namespace cv;

class WeakClassifier {

private:
	float error;
	int dimension;
	float threshold;
	float alpha;
	float beta;
	example sign;
	int misclassified;
	vector<Rect> whites;
	vector<Rect> blacks;

public:
	WeakClassifier();
	int predict(Data* x);
	int predict(float value);
	int predict(const vector<float>& x);
	float evaluateError(vector<Data*>& features);
	void printInfo();
	float getError() const;
	void setError(float error);
	int getDimension() const;
	void setDimension(int dimension);
	float getThreshold() const;
	void setThreshold(float threshold);
	float getAlpha() const;
	void setAlpha(float alpha);
	example getSign() const;
	void setSign(example sign);
	int getMisclassified() const;
	void setMisclassified(int misclassified);
	float getBeta() const;
	void setBeta(float beta);
	const vector<Rect>& getBlacks() const;
	void setBlacks(const vector<Rect>& blacks);
	const vector<Rect>& getWhites() const;
	void setWhites(const vector<Rect>& whites);
	~WeakClassifier() {}
};

#endif
