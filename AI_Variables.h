#pragma once

#include <array>

/*

num_layers: 
	Mindestens 2 (Input - & Output Layer), alle weiteren sind hidden layers

ai_layout: 
	Number of Neurons per Layer

*/ 
constexpr int num_layers = 3; 
constexpr std::array<int, num_layers> ai_layout = {2, 3, 2};

constexpr float ai_learn_factor = 0.5f; 


namespace SimpleAI {

	struct Data_Point {

		std::array<float, ai_layout[0]> data; 
		std::array<float, ai_layout[num_layers - 1]> result; 

	};

	struct Neuron {

		float z = 0; // ohne aktivierungsfunktion
		float a = 0; // mit aktivierungsfunktion 

		float delta_value = 0; // the delta value calculated in backpropagation

	};

	struct Weight {

		float current_weight = 0; // momentaner wert
		float delta_change = 0;   // zukünftige delta Veränderung

		Weight(float w) {
			this->current_weight = w; 
		}

	};


	struct Bias {

		float current_bias = 0; // momentaner wert
		float delta_change = 0;   // zukünftige delta Veränderung

		Bias(float b) {
			this->current_bias = b;
		}

	};

}