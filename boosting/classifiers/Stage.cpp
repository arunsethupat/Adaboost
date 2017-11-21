#include "Stage.h"

Stage::Stage(int number):
    number(number), classifiers({}), fpr(1.), detectionRate(1.), threshold(0.){
}

Stage::Stage(int number, vector<WeakClassifier*> weaks):
    number(number), classifiers(weaks), fpr(1.), detectionRate(1.){
	threshold = 0;
	for(int i = 0; i < classifiers.size(); ++i){
		threshold += classifiers[i]->getAlpha();
	}
}

int Stage::predict(const vector<float>& x){
	float sum = 0;
	int prediction;
	for (int i = 0; i < classifiers.size(); ++i) {
		prediction = classifiers[i]->predict(x) == 1 ? 1 : 0;
		sum += classifiers[i]->getAlpha() * prediction;
	}
	return sum >= threshold ? 1 : 0;
}

int Stage::predict(Mat img){
	float sum = 0;
	float value;
	int prediction;
	for (int i = 0; i < classifiers.size(); ++i) {
		value = HaarFeatures::evaluate(img, classifiers[i]->getWhites(), classifiers[i]->getBlacks());
		prediction = classifiers[i]->predict(value) == 1 ? 1 : 0;
		sum += classifiers[i]->getAlpha() * prediction;
	}
	return sum >= threshold ? 1 : 0;
}


void Stage::optimizeThreshold(vector<Data*>& positiveSet, float dr){
	cout << "Optimizing threshold for stage" << endl;
	vector<float> scores(positiveSet.size());
	float thr;
	int prediction;
	for(int i = 0; i < positiveSet.size(); ++i){
		scores[i] = 0;
		for(int j = 0; j < classifiers.size(); ++j){
			prediction = classifiers[j]->predict(positiveSet[i]) == 1 ? 1 : 0;
			scores[i] += classifiers[j]->getAlpha() * prediction;
		}
	}
	sort(scores.begin(), scores.end());
	int index = positiveSet.size() - dr * positiveSet.size();
	if(index >= 0 && index < positiveSet.size()){
		thr = scores[index];
		if(thr == 0){
			while(index < positiveSet.size() - 1 && scores[index] == 0){
				index++;
			}
			thr = scores[index];
		}
		threshold = scores[index];
	}
	cout << "Setting threshold to " << threshold << endl;
	scores.clear();
}

void Stage::decreaseThreshold(){
	float value = 1;
	threshold -= value;
	cout << "Decrease threshold to " << threshold << endl;
}

float Stage::getThreshold() const {
	return threshold;
}

void Stage::setThreshold(float threshold) {
	this->threshold = threshold;
}

float Stage::getDetectionRate() const {
	return detectionRate;
}

void Stage::setDetectionRate(float detectionRate) {
	this->detectionRate = detectionRate;
}

float Stage::getFpr() const {
	return fpr;
}

void Stage::setFpr(float fpr) {
	this->fpr = fpr;
}

int Stage::getNumber() const {
	return number;
}

void Stage::setClassifiers(const vector<WeakClassifier*>& classifiers) {
	this->classifiers = classifiers;
	threshold = 0;
	for (int i = 0; i < classifiers.size(); ++i) {
		threshold += classifiers[i]->getAlpha();
	}
	threshold = threshold * 0.5;
}

void Stage::setNumber(int number) {
	this->number = number;
}

void Stage::printInfo(){
	cout << "\nStage n. " << number << ", FPR: " << fpr << ", DR: " << detectionRate << ", Threshold: " << threshold << endl;
}

const vector<WeakClassifier*>& Stage::getClassifiers() const {
	return classifiers;
}

void Stage::addClassifier(WeakClassifier* wc){
	classifiers.push_back(wc);
}

Stage::~Stage() {
}

