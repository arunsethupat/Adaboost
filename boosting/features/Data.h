#ifndef FEATURE_DATA_H_
#define FEATURE_DATA_H_

#include <vector>
#include <string>
#include <iostream>

using namespace std;

class Data {
private:
	vector<float> features;
	int label;
	int clas;
	float weight;

public:
	Data(vector<float> features);
	Data(vector<float> features, int label);
	Data(vector<float> features, int label, int clas);
	~Data();
	void print();
	const vector<float>& getFeatures() const;
	void setFeatures(const vector<float>& features);
	int getLabel() const;
	void setLabel(int label);
	float getWeight() const;
	void setWeight(float weight);
	int getClas() const;
	void setClas(int clas);
};

#endif
