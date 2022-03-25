#pragma once

#include <array>

/*

num_layers: 
	Mindestens 2 (Input - & Output Layer), alle weiteren sind hidden layers

ai_layout: 
	Number of Neurons per Layer

*/ 

#define DATA_TYPE float

constexpr int num_layers = 4; 
constexpr std::array<int, num_layers> ai_layout = {2, 3, 4, 1};

constexpr DATA_TYPE ai_learn_factor = 0.01f;


namespace SimpleAI {

	struct Data_Point {

		std::array<DATA_TYPE, ai_layout[0]> data;
		std::array<DATA_TYPE, ai_layout[num_layers - 1]> result;

	};

	struct Neuron {

		DATA_TYPE z = 0; // ohne aktivierungsfunktion
		DATA_TYPE a = 0; // mit aktivierungsfunktion 

		DATA_TYPE delta_value = 0; // the delta value calculated in backpropagation

	};

	struct Weight {

		DATA_TYPE current_weight = 0; // momentaner wert
		DATA_TYPE delta_change = 0;   // zuk�nftige delta Ver�nderung

		Weight(DATA_TYPE w) {
			this->current_weight = w; 
		}

	};


	struct Bias {

		DATA_TYPE current_bias = 0; // momentaner wert
		DATA_TYPE delta_change = 0;   // zuk�nftige delta Ver�nderung

		Bias(DATA_TYPE b) {
			this->current_bias = b;
		}

	};

}