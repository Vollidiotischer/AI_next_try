#pragma once


namespace SimpleAI {

	enum class Layers {
		DENSE,

	};


	struct NeuronLayer {

		NeuronLayer(Eigen::VectorXd neurons, std::function<double(double)> act_fun) : vect(neurons), activation_func(act_fun){}

		Eigen::VectorXd vect; 
		std::function<double(double)> activation_func; 

	};

}