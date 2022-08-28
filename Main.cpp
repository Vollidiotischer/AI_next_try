#include <iostream>

#include <mutex>

#include "SimpleAI/SimpleAI.h"

#include "NeuralNetwork.hpp"

#include "Variables.h"
#include "MainLoop.h"



int main() {

	SimpleAI::NeuralNetwork nn({
		{SimpleAI::Layers::DENSE, 3},
		//{SimpleAI::Layers::DENSE, 3},
	});

	nn.set_error_fun(); 

	return 0; 
}

/*

Instance erschaffen
	-> Activation methode für jedes Layer festlegen 
	-> Error funktion festlegen (funktion als lambda reingeben) 
	-> Lernmethode festlegen 

Genetic Algorithm: 
	
*/