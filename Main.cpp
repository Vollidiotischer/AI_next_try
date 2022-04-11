#include <iostream>

#include <mutex>

#include "SimpleAI/SimpleAI.h"

#include "Variables.h"
#include "MainLoop.h"

/*

todo: 
	increase performance (remove all the checks / debug mode with #ifdef)
		+ remove all the unneseccary vector copying


Training data format: 
	First Line: Input size & output size seperated by a colon (:)
	Following lines: Data
		Data is seperated by a dot (.)
		At the end of each line is the expected result
			It is seperated from the data by a dash (-)





Berechnen: 
	Ergebnis des NN 
	a: Ergebnis aller Zwischenneuronen (o in WWW)
	z: Ergebnis aller Neuronen ohne aktivierungsfunktion (a in WWW)
*/



static void apply_softmax_function(std::vector<float>& inout) {

	DATA_TYPE sum = 0;


	DATA_TYPE max = inout[0];

	for (int i = 0; i < inout.size(); i++) {
		if (inout[i] > max) {
			max = inout[i];
		}
	}

	if (max > 10000) {
		std::cout << max << std::endl;
	}

	for (int i = 0; i < inout.size(); i++) {
		inout[i] -= max;
		sum += std::exp(inout[i]);
	}

	for (int i = 0; i < inout.size(); i++) {
		inout[i] = (std::exp(inout[i]) / sum);
	}


}

static void apply_softmax_function2(std::vector<float>& inout) {

	float sum0 = 0; 

	for (int i = 0; i < inout.size(); i++) {
		sum0 += (inout[i]);
	}

	for (int i = 0; i < inout.size(); i++) {
		inout[i] = ((inout[i]) / sum0);
	}

	float sum = 0; 

	for (int i = 0; i < inout.size(); i++) {
		sum += std::exp(inout[i]);
	}

	for (int i = 0; i < inout.size(); i++) {
		//inout[i] = (std::exp(inout[i]) / sum);
	}


}

int main() {
	/*
	std::vector<float> a = { 10, 15 }; 
	apply_softmax_function(a); 
	SimpleAI::println_vector(a, 0); 



	std::vector<float> a2 = { 10, 15 };
	apply_softmax_function2(a2);
	SimpleAI::println_vector(a2, 0);
	*/

	SimpleAI::AI_Manager manager(10); 

	std::vector<Point> points;


	MainLoop::run(manager, points); 


	return 0; 
}

