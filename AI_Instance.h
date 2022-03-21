#pragma once

#include <stdio.h>

#include <array>
#include <vector>
#include <random>
#include <time.h>

#include "AI_Variables.h"


namespace SimpleAI {

	namespace {
		struct Random_Device {
			std::default_random_engine engine;
			std::normal_distribution<float> dist;

			Random_Device() : dist(0.f, 1.f), engine(time(NULL)) {}

			float get_random_number() {

				return dist(engine);
			}

		};
	}


	struct AI_Instance {

		static Random_Device rd;

		/*

		Weights is a List of num_layers - 1 matrices

		*/
		std::array<std::vector<std::vector<float>>, num_layers - 1> weights;
		std::array<std::vector<float>, num_layers - 1> biases;
		std::array<std::vector<float>, num_layers> neurons;


		AI_Instance() {

			// initialize neurons
			for (int i = 0; i < num_layers; i++) {
				neurons[i].resize(ai_layout[i]);
			}


			// initialize weights
			for (int i = 0; i < weights.size(); i++) {
				// number of rows in the matrix (one row for each neuron in the following layer) 
				weights[i] = std::vector<std::vector<float>>(ai_layout[i + 1], std::vector<float>(ai_layout[i], 0));

				// Initialize weights to random value
				for (int i2 = 0; i2 < weights[i].size(); i2++) {
					for (int i3 = 0; i3 < weights[i][i2].size(); i3++) {
						weights[i][i2][i3] = rd.get_random_number();
					}
				}

			}

			// initialize biases
			for (int i = 0; i < biases.size(); i++) {
				biases[i] = std::vector<float>(ai_layout[i + 1], 0);

				// Initialize to random Value
				for (int i2 = 0; i2 < biases[i].size(); i2++) {
					biases[i][i2] = rd.get_random_number();
				}
			}
		}


		void evaluate_input(const std::vector<float>& data /* IN */, std::vector<float>& result /* OUT */) {

			if (data.size() != neurons[0].size()) {
				fprintf(stderr, "Data size (%d) does not match Input Layer size (%d) [Line %d in function '%s']", data.size(), neurons[0].size(), __LINE__, __func__);
			}

			// copy data to neurons (may be optimised out)
			for (int i2 = 0; i2 < neurons[0].size(); i2++) {
				neurons[0][i2] = data[i2];
			}

			// Feed data to AI
			for (int i = 0; i < num_layers - 1; i++) {
				zero_out(neurons[i + 1]);
				matrix_vector_multiply(neurons[i], weights[i], neurons[i + 1]);
				vect_vect_add(neurons[i + 1], biases[i]);
				apply_activation_function(neurons[i + 1]);
			}

			// Copy elements from last neuron layer to result vector (may be optimised out)
			result.resize(neurons.back().size());

			for (int i = 0; i < result.size(); i++) {
				result[i] = neurons.back()[i];
			}

		}

	private:

		static float activation_function(float x) {
			return ((0.5f * x) / (1.f + abs(x))) + 0.5f;
		}

		static void zero_out(std::vector<float>& vect1 /* IN / OUT */) {
			for (int i = 0; i < vect1.size(); i++) {
				vect1[i] = 0.f;
			}
		}

		static void matrix_vector_multiply(const std::vector<float>& vect /* IN */, const std::vector<std::vector<float>>& matrix /* IN */, std::vector<float>& result /* OUT */) {

			if (matrix.size() != result.size()) {
				fprintf(stderr, "Matrix size (%d) does not match Result size (%d) [Line %d in function '%s']", matrix.size(), result.size(), __LINE__, __func__);
			}

			for (int i = 0; i < matrix.size(); i++) {

				if (matrix[i].size() != vect.size()) {
					fprintf(stderr, "Matrix[i] size (%d) does not match Vect size (%d) [Line %d in function '%s']", matrix[i].size(), vect.size(), __LINE__, __func__);
				}

				for (int i2 = 0; i2 < matrix[i].size(); i2++) {
					result[i] += matrix[i][i2] * vect[i2];
				}

			}

		}

		static void vect_vect_add(std::vector<float>& vect1 /* IN / OUT */, const std::vector<float>& vect2 /* IN */) {

			if (vect1.size() != vect2.size()) {
				fprintf(stderr, "Vector sizes do not match (%d and %d) [Line %d in function '%s']", vect1.size(), vect2.size(), __LINE__, __func__);
			}

			for (int i = 0; i < vect1.size(); i++) {
				vect1[i] += vect2[i];
			}
		}

		static void apply_activation_function(std::vector<float>& vect /*IN / OUT */) {

			for (int i = 0; i < vect.size(); i++) {
				vect[i] = activation_function(vect[i]);
			}

		}
	};

	// Initialize static Random_Device
	Random_Device AI_Instance::rd;

}
