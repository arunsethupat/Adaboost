#ifndef BOOSTING_CLASSIFIERS_STAGE_H_
#define BOOSTING_CLASSIFIERS_STAGE_H_

#include "WeakClassifier.h"
#include <vector>
#include <iostream>
#include "../features/HaarFeatures.h"

using namespace std;

class Stage {
private:
	int number;
	vector<WeakClassifier*> classifiers;
	float fpr;
	float detectionRate;
	float threshold;

public:
	Stage(int number);
	Stage(int number, vector<WeakClassifier*> weaks);
	int predict(const vector<float>& x);
	int predict(Mat img);
	void decreaseThreshold();
	void optimizeThreshold(vector<Data*>& positiveSet, float dr);
	void printInfo();
	float getThreshold() const;
	void setThreshold(float threshold);
	float getDetectionRate() const;
	void setDetectionRate(float detectionRate);
	float getFpr() const;
	void setFpr(float fpr);
	int getNumber() const;
	void setNumber(int number);
	const vector<WeakClassifier*>& getClassifiers() const;
	void setClassifiers(const vector<WeakClassifier*>& classifiers);
	void addClassifier(WeakClassifier* wc);
	~Stage();
};

#endif
