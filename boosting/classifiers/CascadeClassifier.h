
#ifndef BOOSTING_CLASSIFIERS_CASCADECLASSIFIER_H_
#define BOOSTING_CLASSIFIERS_CASCADECLASSIFIER_H_

#include <vector>
#include "Stage.h"

using namespace std;

class CascadeClassifier {
private:
	vector<Stage*> stages;

public:
	CascadeClassifier();
	void addStage(Stage* stage);
	void train();
	int predict(Mat img);
	int predict(const vector<float>& x);
	float score(const vector<float>& x);
	~CascadeClassifier();
	const vector<Stage*>& getStages() const;
	void setStages(const vector<Stage*>& stages);
};

#endif
