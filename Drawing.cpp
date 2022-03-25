#include <iostream>

#include <vector>

#include <SFML/Graphics.hpp>

#include "SimpleAI.h"

#include "Variables.h"

namespace Drawing {

	namespace {

		void paint_background(sf::RenderWindow& rw, SimpleAI::AI_Instance* ai) {

			if (ai == NULL) {
				return; 
			}

			// draw background
			sf::RectangleShape rect({point_size, point_size});

			for (int i = 0; i < window_width / point_size; i++) {

				for (int i2 = 0; i2 < window_height / point_size; i2++) {

					rect.setPosition({i * point_size, i2 * point_size});

					std::array<DATA_TYPE, 1> result;
					std::array<DATA_TYPE, 2> data = { i, i2 };
					ai->evaluate_input(data, result);
					rect.setFillColor(sf::Color(result[0] * 255, 0, (1.f - result[0]) * 255));

					rw.draw(rect); 
				}

			}


		}

		void paint_points(sf::RenderWindow& rw, std::vector<Point>& points) {

			// draw points
			sf::CircleShape circle(circle_radius, 10);
			circle.setOutlineThickness(2);
			circle.setOutlineColor(sf::Color::Black);
			circle.setOrigin({ circle_radius, circle_radius });

			for (int i = 0; i < points.size(); i++) {
				circle.setPosition({ (float)points[i].x, (float)points[i].y });
				circle.setFillColor(points[i].color == 'r' ? sf::Color::Red : sf::Color::Blue);

				rw.draw(circle);
			}
		}

	}

	void draw_screen(sf::RenderWindow& rw, std::vector<Point>& points, SimpleAI::AI_Instance* ai) {

		rw.clear(sf::Color::White); 

		paint_background(rw, ai); 

		paint_points(rw, points); 

		rw.display(); 

	}

}