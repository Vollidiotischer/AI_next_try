
#include <stdio.h>
#include <iostream>
#include <array>
#include <vector>
#include <random>
#include <time.h>
#include <functional>

#include "SimpleAI.h"


namespace SimpleAI {

	AI_Instance::AI_Instance(double learn_factor) {

		this->learn_factor = learn_factor;

		// initialize neurons
		for (int i = 0; i < num_layers; i++) {
			neurons[i].resize(ai_layout[i]);
		}


		// initialize weights
		init_weights();

		// initialize biases
		init_biases();
	}


	void AI_Instance::print_instance() {
		std::cout << "Status of the ai: " << std::endl;
		for (int i = 0; i < neurons.size(); i++) {

			std::cout << "a: [";

			for (int i2 = 0; i2 < neurons[i].size(); i2++) {
				std::cout << neurons[i][i2].a << (i2 < neurons[i].size() - 1 ? ", " : "]");
			}
			std::cout << std::endl;

			std::cout << "z: [";

			for (int i2 = 0; i2 < neurons[i].size(); i2++) {
				std::cout << neurons[i][i2].z << (i2 < neurons[i].size() - 1 ? ", " : "]");
			}

			std::cout << std::endl;

			std::cout << "d: [";

			for (int i2 = 0; i2 < neurons[i].size(); i2++) {
				std::cout << neurons[i][i2].delta_value << (i2 < neurons[i].size() - 1 ? ", " : "]");
			}
			std::cout << std::endl << std::endl;
		}

		std::cout << "Error of the ai: " << error << std::endl << std::endl;

		std::cout << "Weights of the ai: " << std::endl;

		std::cout << "Weights_0: [" << std::endl;
		for (auto& s1 : weights) {
			std::cout << "\tWeights_1: [" << std::endl;
			for (auto& s2 : s1) {
				SimpleAI::println_vector_weight(s2, 2);
			}
			std::cout << "\t]" << std::endl;
		}
		std::cout << "]" << std::endl << std::endl;

		std::cout << "Biases of the ai: " << std::endl;

		std::cout << "Biases_0: [" << std::endl;

		for (auto& s1 : biases) {
			SimpleAI::println_vector_bias(s1, 1);
		}

		std::cout << "]" << std::endl;
	}


	void AI_Instance::print_error(std::string app = "") {
		std::cout << "Error: " << error * 100.f << "%" << app;
	}


	void AI_Instance::init_weights() {
		for (int i = 0; i < weights.size(); i++) {
			// number of rows in the matrix (one row for each neuron in the following layer) 
			weights[i] = std::vector<std::vector<Weight>>(ai_layout[i + 1], std::vector<Weight>(ai_layout[i], 0));

			// Initialize weights to random value
			for (int i2 = 0; i2 < weights[i].size(); i2++) {
				for (int i3 = 0; i3 < weights[i][i2].size(); i3++) {
					weights[i][i2][i3].current_weight = rd.get_random_number() / sqrt((double)neurons[i].size());
				}
			}

		}
	}

	void AI_Instance::init_biases() {
		for (int i = 0; i < biases.size(); i++) {
			biases[i] = std::vector<Bias>(ai_layout[i + 1], 0);

			// Initialize to random Value
			for (int i2 = 0; i2 < biases[i].size(); i2++) {
				biases[i][i2].current_bias = rd.get_random_number();
			}
		}
	}

	void AI_Instance::clear_weight_delta_value(std::array<std::vector<std::vector<Weight>>, num_layers - 1>& weights) {

		for (auto& w1 : weights) {
			for (auto& w2 : w1) {
				for (auto& w3 : w2) {
					w3.delta_change = 0.f;
				}
			}
		}

	}


	void AI_Instance::clear_biases_delta_value(std::array<std::vector<Bias>, num_layers - 1>& biases) {

		for (auto& b1 : biases) {
			for (auto& b2 : b1) {
				b2.delta_change = 0.f;
			}
		}

	}

	void AI_Instance::evaluate_input_list(AI_Instance& ai, std::vector<Data_Point>& data) {

		ai.error = 0.f;

		for (int i = 0; i < data.size(); i++) {
			// Feed data through AI
			feed_forward_step(ai, data[i]);

		}

		// average out the error
		ai.error /= (double)data.size();

	}

	void AI_Instance::feed_forward_step(AI_Instance& ai, Data_Point& dp) {

		std::array<double, ai_layout[num_layers - 1]> result;
		evaluate_input(ai, dp.data, result); // static 

		ai.error += cost_function(result, dp.result); // static

	}
}