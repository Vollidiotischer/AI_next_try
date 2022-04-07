#include <iostream>

#include <vector>

#include <SFML/Graphics.hpp>

#include "SimpleAI/SimpleAI.h"

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

					std::array<DATA_TYPE, 2> result;
					std::array<DATA_TYPE, 2> data = { (DATA_TYPE)i / (window_width/point_size), (DATA_TYPE)i2 / (window_height/point_size) };
					ai->evaluate_input(data, result);
					
					rect.setFillColor(sf::Color(result[0] * 255, 0, result[1] * 255));


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
				circle.setFillColor(sf::Color((points[i].color == 'r') * 255, (points[i].color == 'g') * 255, (points[i].color == 'b') * 255));

				rw.draw(circle);
			}
		}

	}

	void draw_screen(sf::RenderWindow& rw, std::vector<Point>& points, SimpleAI::AI_Instance* ai, bool points_shown) {

		rw.clear(sf::Color::White); 

		paint_background(rw, ai); 

		if (points_shown)
			paint_points(rw, points); 

		rw.display(); 

	}

}