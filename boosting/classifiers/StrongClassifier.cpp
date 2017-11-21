
#include "StrongClassifier.h"
StrongClassifier::StrongClassifier(vector<WeakClassifier*> classifiers):
	classifiers(classifiers){}

int StrongClassifier::predict(Data* x){
	float sum = 0;
	for(int i = 0; i < classifiers.size(); ++i){
		sum += classifiers[i]->getAlpha() * classifiers[i]->predict(x);
	}
	if(sum > 0){
		return 1;
	} else {
		return 0;
	}
}

StrongClassifier::~StrongClassifier(){
	classifiers.clear();
}

const vector<WeakClassifier*>& StrongClassifier::getClassifiers() const {
	return classifiers;
}

void StrongClassifier::setClassifiers(const vector<WeakClassifier*>& classifiers) {
	this->classifiers = classifiers;
}
