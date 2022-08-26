#include <iostream>


#include "SimpleAI/SimpleAI.h"
#include "Variables.h"


namespace AIModule {

	namespace {

		void populate_data(std::vector<SimpleAI::Data_Point>& data, std::vector<Point>& points) {

			for (auto& p : points) {
				SimpleAI::Data_Point dp;
				dp.data[0] = (double)p.x / window_width;
				dp.data[1] = (double)p.y / window_height;

				dp.result[0] = p.color == 'r';
				dp.result[1] = p.color == 'b';
				//dp.result[2] = p.color == 'b';

				data.push_back(dp);
			}

		}

	}

	void train_ai(std::vector<Point>& points, SimpleAI::AI_Manager& manager) {

		static int i = 0;
		i++;

		std::vector<SimpleAI::Data_Point> data;

		populate_data(data, points);

		manager.train_all_instances(data, 0, data.size());

		manager.evaluate_instances(data);

		manager.calculate_best_instance();
		
		if (i % 100 == 0) {

			manager.reshuffel_instances();
			manager.best_instance->print_error("\n");

			i = 0;
		}

		for (int index = 0; index < manager.ai_list.size(); index++) {
			if (std::isnan(manager.ai_list[index].error) || std::isnan(-manager.ai_list[index].error)) {
				manager.ai_list[index] = SimpleAI::AI_Instance(SimpleAI::ai_learn_factor); 
			}
		}

	}
}