#ifndef BOOSTING_VIOLAJONES_H_
#define BOOSTING_VIOLAJONES_H_

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "AdaBoost.h"
#include "classifiers/StrongClassifier.h"
#include "classifiers/CascadeClassifier.h"
#include "features/HaarFeatures.h"
#include "utils/Face.h"
#include "utils/Utils.hpp"

using namespace std;
using namespace cv;

class ViolaJones: public AdaBoost {
private:
	int maxStages;
	vector<Data*> positives;
	vector<Data*> negatives;
	vector<Data*> validation;
	string positivePath;
	string negativePath;
	string validationPath;
	int numPositives;
	int numNegatives;
	int numValidation;
	bool useNormalization;
	CascadeClassifier classifier;
	int negativesPerLayer;
	int detectionWindowSize;
	float evaluateFPR(vector<Data*>& validationSet);
	float evaluateDR(vector<Data*>& validationSet);
	void optimizeThreshold(vector<Data*>& positiveSet, float dr);
	void initializeWeights();
	void generateNegativeSet(int number, bool rotate);
	void extractFeatures();

protected:
	float updateAlpha(float error);
	float updateBeta(float error);
	void updateWeights(WeakClassifier* weakClassifier);

public:
	ViolaJones();
	ViolaJones(string trainedPath);
	ViolaJones(string positivePath, string negativePath, int maxStages, int numPositives,
			int numNegatives, int detectionWindowSize = 24, int negativesPerLayer = 0);
	vector<Face> mergeDetections(vector<Face>& detections, int padding = 6, float th = 0.5);
	void train();
	int predict(Mat img);
	void loadTrainedData(string filename);
	const string& getValidationPath() const;
	void setValidationSet(const string& validationPath, int examples = -1);
	void store();
	void normalizeImage(Mat& img);
	const CascadeClassifier& getClassifier() const;
	void setClassifier(const CascadeClassifier& classifier);
	int getMaxStages() const;
	void setMaxStages(int maxStages);
	const string& getNegativePath() const;
	void setNegativePath(const string& negativePath);
	int getNegativesPerLayer() const;
	void setNegativesPerLayer(int negativesPerLayer);
	int getNumNegatives() const;
	void setNumNegatives(int numNegatives);
	int getNumPositives() const;
	void setNumPositives(int numPositives);
	const string& getPositivePath() const;
	void setPositivePath(const string& positivePath);
	bool isUseNormalization() const;
	void setUseNormalization(bool useNormalization);
	~ViolaJones();
};

#endif
