#pragma once

#include <vector>


namespace SimpleAI {

	struct AI_Manager {

		int num_instances; 

		std::vector<AI_Instance> ai_list; 

		AI_Manager(int num_instances) : num_instances(num_instances) {

			std::cout << "[AI_Manager] Creating instances..." << std::endl; 

			for (int i = 0; i < num_instances; i++) {
				ai_list.push_back(AI_Instance()); 
			}

			std::cout << "[AI_Manager] Finished creating Instances" << std::endl; 

		}

		void train_all_instances(std::vector<Data_Point>& data, int iterations, int print_interval = 0) {

			if (print_interval == 0) {
				this->train_all_instances_without_print(data, iterations);
				return; 
			}

			for (int i = 0; i < iterations; i++) {
				for (int i2 = 0; i2 < ai_list.size(); i2++) {
					ai_list[i2].backprop(data);
				}

				if (i % print_interval == 0) {
					std::cout << i / print_interval << "/" << iterations / print_interval << " (In " << print_interval << " Steps)" << std::endl;
				}
			}

		}


		void train_all_instances(std::vector<Data_Point>& data, float goal_percent) {

			goal_percent /= 100.f;

			float minimum = 1.f; 

			while (minimum > goal_percent){
				for (int i2 = 0; i2 < ai_list.size(); i2++) {
					ai_list[i2].backprop(data);
					if (ai_list[i2].error < minimum) {
						minimum = ai_list[i2].error;
						std::cout.precision(4); 
						std::cout << "Error: " << minimum * 100.f << "% / " << goal_percent * 100.f << "%" << std::endl;

					} 
				}
			}

		}

		void evaluate_instances(std::vector<Data_Point>& data) {

			for (int i = 0; i < num_instances; i++) {

				// evaluate data for each ai instance, internally set error variable
				ai_list[i].evaluate_input_list(data); 

			}

		}

	private:
		
		void train_all_instances_without_print(std::vector<Data_Point>& data, int iterations) {
		
			for (int i = 0; i < iterations; i++) {
				for (int i2 = 0; i2 < ai_list.size(); i2++) {
					ai_list[i2].backprop(data);
				}
			}
		
		}

	};

}