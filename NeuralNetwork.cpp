
#include <iostream>
#include <Eigen/Dense>

#include "NeuralNetwork.hpp"

namespace SimpleAI {

	NeuralNetwork::NeuralNetwork(std::vector<std::tuple<Layers, int, std::function<double(double)>>> layers) {

		for (auto& t : layers) {
			this->neurons.push_back(
				NeuronLayer(
					Eigen::VectorXd(std::get<1>(t)), 
					std::get<2>(t)
				)
			); 
			
		}

		if (layers.size() < 2) {
			std::cerr << "NeuralNetwork of size " << layers.size() << " is too small (min size 2 for in/out layers)" << std::endl; 
			exit(1); 
		}

		this->initialize_weights(); 
		this->initialize_biases(); 

	}

	void NeuralNetwork::initialize_weights() {

		// go through neuron pairs (n1, n2)
		// Matrix is of size (n2 x n1)

		for (int i = 0; i < this->neurons.size() - 1; i++) {
			auto& n1 = this->neurons[i]; 
			auto& n2 = this->neurons[i + 1];

			Eigen::MatrixXd weight_matrix(n2.vect.size(), n1.vect.size());

			this->weights.push_back(weight_matrix); 

		}

	}


	void NeuralNetwork::initialize_biases() {

		// Go through all layers starting from the second
		// create a bias vector the same size as the weight vector

		for (int i = 1; i < this->neurons.size(); i++) {

			Eigen::VectorXd bias_vect(this->neurons[i].vect.size());
			this->biases.push_back(bias_vect); 

		}

	}

	void NeuralNetwork::randomize_weights() {

	}
	
	void NeuralNetwork::randomize_biases() {

	}

}