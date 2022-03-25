#include <iostream>

#include <vector>

#include <SFML/Graphics.hpp>

#include "SimpleAI.h"

#include "Variables.h"

namespace Events {

	void handle_events(sf::RenderWindow& rw, std::vector<Point>& points, SimpleAI::AI_Manager& manager) {

		sf::Event events; 
		while (rw.pollEvent(events)) {

			if (events.type == sf::Event::Closed) {
				rw.close(); 
			}

			// create new point
			if (events.type == sf::Event::MouseButtonPressed) {

				int mx = sf::Mouse::getPosition(rw).x; 
				int my = sf::Mouse::getPosition(rw).y; 

				if (events.mouseButton.button == sf::Mouse::Left) {
					points.push_back(Point(mx, my, 'r')); 
				}

				if (events.mouseButton.button == sf::Mouse::Right) {
					points.push_back(Point(mx, my, 'b'));

				}

			}

			// train ai
			if (events.type == sf::Event::KeyPressed) {
				if (events.key.code == sf::Keyboard::Space) {

					//train_ai(points, manager);
					manager.calculate_best_instance();

					manager.best_instance->print_instance();

				}
			}

		}

	}

}