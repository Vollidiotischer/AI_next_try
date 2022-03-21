#pragma once


#include <vector>


namespace SimpleAI {

	struct AI_Manager {

		int num_instances; 

		std::vector<AI_Instance> ai_list; 

		AI_Manager(int num_instances) : num_instances(num_instances) {

			for (int i = 0; i < num_instances; i++) {
				ai_list.push_back(AI_Instance()); 
			}

		}

		// give in training data, an error goal... 
		void train_ai() {
			/*
			Testing: Random values for training
			*/


		}
	};

}