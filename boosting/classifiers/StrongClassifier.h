#ifndef BOOSTING_CLASSIFIERS_STRONGCLASSIFIER_H_
#define BOOSTING_CLASSIFIERS_STRONGCLASSIFIER_H_

#include <vector>
#include <iostream>
#include "../features/Data.h"
#include "WeakClassifier.h"

class StrongClassifier {

protected:
	vector<WeakClassifier*> classifiers;

public:
	StrongClassifier(vector<WeakClassifier*> classifiers);
	int predict(Data* x);
	~StrongClassifier();
	const vector<WeakClassifier*>& getClassifiers() const;
	void setClassifiers(const vector<WeakClassifier*>& classifiers);
};



#endif
