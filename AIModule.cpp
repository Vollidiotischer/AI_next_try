#include <iostream>


#include "SimpleAI.h"
#include "Variables.h"


namespace AIModule {

	namespace {

		void populate_data(std::vector<SimpleAI::Data_Point>& data, std::vector<Point>& points) {

			for (auto& p : points) {
				SimpleAI::Data_Point dp;
				dp.data[0] = p.x / point_size;
				dp.data[1] = p.y / point_size;

				dp.result[0] = p.color == 'r';
				//dp.result[1] = p.color == 'b';

				data.push_back(dp);
			}

		}

	}

	void train_ai(std::vector<Point>& points, SimpleAI::AI_Manager& manager) {

		static int i = 0;
		i++;

		std::vector<SimpleAI::Data_Point> data;

		populate_data(data, points);

		manager.train_all_instances(data, 500, 0);

		manager.evaluate_instances(data);

		manager.calculate_best_instance();

		if (i % 50 == 0) {
			manager.reshuffel_instances();
			manager.best_instance->print_error("\n");

			i = 0;
		}

	}
}