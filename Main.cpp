#include <iostream>

#include "SimpleAI.h"


/*

Training data format: 
	First Line: Input size & output size seperated by a colon (:)
	Following lines: Data
		Data is seperated by a dot (.)
		At the end of each line is the expected result
			It is seperated from the data by a dash (-)

*/

int main() {

	//SimpleAI::AI_Manager manager(1); 

	std::vector<std::string> vect; 
	SimpleAI::Resource_Manager::read_txt_file("training_data.txt", vect); 

	std::vector<float> res; 

	SimpleAI::Resource_Manager::parse_data(vect[1], res); 

	for (auto& s : res) {
		std::cout << s << std::endl; 
	}


	return 0; 
}