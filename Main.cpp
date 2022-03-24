#include <iostream>

#include "SimpleAI.h"


/*

todo: 
	increase performance (remove all the checks / debug mode with #ifdef)
		+ remove all the unneseccary vector copying




Ablauf: 
	Load data
	Evaluate all ai instances
		-> Save error for each instance
	train 
	repeat 




Berechnen: 
	Ergebnis des NN 
	a: Ergebnis aller Zwischenneuronen (o in WWW)
	z: Ergebnis aller Neuronen ohne aktivierungsfunktion (a in WWW)
*/



int main() {

	std::vector<SimpleAI::Data_Point> data;

	SimpleAI::Resource_Manager::load_data("training_data.txt", data);
	
	SimpleAI::AI_Manager manager(1); 

	//manager.train_all_instances(data, 1000000, 10000);
	manager.train_all_instances(data, 1.f);


	for (auto& s : manager.ai_list) {
		std::cout << "\nError: " << s.error << " ==> " << s.error * 100.f << "%" << std::endl;
		for (auto& d : data) {
			std::array<float, 2> res; 
			s.evaluate_input(d.data, res); 
			SimpleAI::println_array<float, 2>(d.result); 
			SimpleAI::println_array<float, 2>(res); 
			std::cout << std::endl; 
		}

	}

	

	return 0; 
}