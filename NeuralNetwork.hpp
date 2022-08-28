#pragma once

#include <vector>
#include <Eigen/Dense>
#include "Types.hpp"

namespace SimpleAI {

	struct NeuralNetwork {
		std::vector<NeuronLayer> neurons; 
		std::vector<Eigen::MatrixXd> weights; 
		std::vector<Eigen::VectorXd> biases; 
		
		NeuralNetwork(std::vector<std::tuple<Layers, int, std::function<double(double)>>> layers);
		
	private:
		void initialize_weights(); 
		void initialize_biases(); 

		void randomize_weights(); 
		void randomize_biases(); 
	};

}

/*

Use: 
NeuralNetwork nn ({

	{ Layers::Dense, 3 },
	{ Layers::Dense, 3 },

})

*/