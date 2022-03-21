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

