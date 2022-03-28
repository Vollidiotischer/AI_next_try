#include <iostream>

#include "SimpleAI.h"

#include "Variables.h"
#include "MainLoop.h"

/*

todo: 
	increase performance (remove all the checks / debug mode with #ifdef)
		+ remove all the unneseccary vector copying


Training data format: 
	First Line: Input size & output size seperated by a colon (:)
	Following lines: Data
		Data is seperated by a dot (.)
		At the end of each line is the expected result
			It is seperated from the data by a dash (-)





Berechnen: 
	Ergebnis des NN 
	a: Ergebnis aller Zwischenneuronen (o in WWW)
	z: Ergebnis aller Neuronen ohne aktivierungsfunktion (a in WWW)
*/



int main() {

	SimpleAI::AI_Manager manager(10); 
	std::vector<Point> points;


	MainLoop::run(manager, points); 

	return 0; 
}

