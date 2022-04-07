#include <iostream>

#include <SFML/Graphics.hpp>

#include "SimpleAI/SimpleAI.h"

#include "Variables.h"
#include "Events.h"
#include "Drawing.h"
#include "AIModule.h"


namespace MainLoop {

	void run(SimpleAI::AI_Manager& manager, std::vector<Point>& points) {

		sf::RenderWindow window(sf::VideoMode(500, 500), "IDFK", sf::Style::Close | sf::Style::Titlebar);
		window.setKeyRepeatEnabled(false); 

		bool show_points = true; 

		while (window.isOpen()) {

			Events::handle_events(window, points, manager, show_points); 

			AIModule::train_ai(points, manager); 

			Drawing::draw_screen(window, points, manager.best_instance, show_points);

		}

	}

}