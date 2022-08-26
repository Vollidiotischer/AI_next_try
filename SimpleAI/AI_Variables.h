#pragma once

#include <array>

/*

num_layers: 
	Mindestens 2 (Input - & Output Layer), alle weiteren sind hidden layers

ai_layout: 
	Number of Neurons per Layer

*/ 


namespace SimpleAI {

	constexpr int num_layers = 6;
	constexpr std::array<int, num_layers> ai_layout = { 2, 6, 6, 6, 6, 2 };

	constexpr double ai_learn_factor = 0.01f;

	constexpr float erwartungswert = 0.f;
	constexpr float standardabweichung = 3.f;


	struct Data_Point {

		std::array<double, ai_layout[0]> data;
		std::array<double, ai_layout[num_layers - 1]> result;

	};

	struct Neuron {

		double z = 0; // ohne aktivierungsfunktion
		double a = 0; // mit aktivierungsfunktion 

		double delta_value = 0; // the delta value calculated in backpropagation

	};

	struct Weight {

		double current_weight = 0; // momentaner wert
		double delta_change = 0;   // zukünftige delta Veränderung

		Weight(double w) {
			this->current_weight = w; 
		}

	};


	struct Bias {

		double current_bias = 0; // momentaner wert
		double delta_change = 0;   // zukünftige delta Veränderung

		Bias(double b) {
			this->current_bias = b;
		}

	};

}