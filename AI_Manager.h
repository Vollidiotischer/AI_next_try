#pragma once

#include <vector>


namespace SimpleAI {

	struct AI_Manager {

		int num_instances; 

		std::vector<AI_Instance> ai_list; 

		AI_Instance *best_instance = NULL; 

		AI_Manager(int num_instances) : num_instances(num_instances) {

			srand(time(NULL)); 
			std::cout << "[AI_Manager] Creating instances..." << std::endl; 

			for (int i = 0; i < num_instances; i++) {
				ai_list.push_back(AI_Instance(ai_learn_factor)); 
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


		void train_all_instances(std::vector<Data_Point>& data, DATA_TYPE goal_percent) {

			goal_percent /= 100.f;

			DATA_TYPE minimum = 1.f; 

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

		void reshuffel_instances() {

			int pos = calculate_best_instance(); 

			for (int i = 0; i < ai_list.size(); i++) {
				if (i != pos) {
					ai_list[i].init_biases(); 
					ai_list[i].init_weights();
				}
			}
		}

		void evaluate_instances(std::vector<Data_Point>& data) {

			for (int i = 0; i < num_instances; i++) {

				// evaluate data for each ai instance, internally set error variable
				ai_list[i].evaluate_input_list(data); 

			}

		}

		int calculate_best_instance() {

			int pos = 0; 
			DATA_TYPE min_score = ai_list[0].error; 
			best_instance = &ai_list[0]; 

			for (int i = 1; i < ai_list.size(); i++) {
				if (ai_list[i].error < min_score) {
					min_score = ai_list[i].error;
					best_instance = &ai_list[i];

					pos = i; 

				}
			}

			return pos; 
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