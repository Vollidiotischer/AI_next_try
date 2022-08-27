#pragma once

#include <vector>
#include <Eigen/Dense>
#include "Types.hpp"

namespace SimpleAI {

	struct NeuralNetwork {
		std::vector<Eigen::VectorXd> neurons; 
		std::vector<Eigen::MatrixXd> weights; 
		std::vector<Eigen::VectorXd> biases; 
		
		NeuralNetwork(std::vector<std::pair<Layers, int>> layers); 
		
	private:
		void initialize_weights(); 
		void initialize_biases(); 

	};

}

/*

Use: 
NeuralNetwork nn ({

	Layers::Dense(3),
	Layers::Dense(3),

})

*/