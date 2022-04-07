#pragma once

#include <stdio.h>

#include <vector>
#include <array>
#include <fstream>
#include <string>

namespace SimpleAI {

	struct Resource_Manager {

		static void load_data(std::string path, std::vector<Data_Point>& data) {

			printf("Loading Data...\n"); 

			std::vector<std::string> parsed; 
			read_txt_file(path, parsed); 

			std::array<DATA_TYPE, 2> ai_size;
			parse_data<2>(parsed[0], ai_size); 

			for (int i = 1; i < parsed.size(); i++) {
				data.push_back(Data_Point()); 
				parse_data(parsed[i], i, data[i - 1]); 
			}

			printf("Finished loading Data\n"); 
		}

		static void read_txt_file(std::string& path, std::vector<std::string>& result) {

			std::ifstream file; 
			file.open(path); 

			if (file.is_open()) {

				std::string temp_str; 

				while (std::getline(file, temp_str)) {

					if (temp_str.size() == 0) { continue;  }

					result.push_back(temp_str); 
				}

				file.close();
			}
			else {
				fprintf(stderr, "Text File '%s' could not be opened [Line %d in function '%s']", path.c_str(), __LINE__, __func__); 
				exit(1); 
			}

		}

		template <int arr_size>
		static void parse_data(const std::string& inp /* IN */, std::array<DATA_TYPE, arr_size>& result /* OUT */) {
			
			int last_pos = 0; 

			int data_index = 0; 

			for (int i = 0; i < inp.size(); i++) {
				if (inp[i] == ':') {
					result[0] = stof(inp.substr(0, i)); 
					result[1] = stof(inp.substr(i + 1));
					return; 
				}

				if (inp[i] == '.' || inp[i] == '-' || inp[i] == '!') {
					result[data_index] = stod(inp.substr(last_pos, i - last_pos));
					data_index++; 
					last_pos = i + 1; 
				}

			}
		}

		static void parse_data(const std::string& inp /* IN */, int line, SimpleAI::Data_Point& result /* OUT */) {
			
			int dash_pos = inp.find_first_of('-'); 
			int colon_pos = inp.find_first_of(':'); 

			if (dash_pos != std::string::npos && colon_pos == std::string::npos) {
				parse_data<ai_layout[0]>(inp.substr(0, dash_pos + 1), result.data); 
				parse_data<ai_layout[num_layers - 1]>(inp.substr(dash_pos + 1), result.result);
			}
			 
			if (dash_pos == std::string::npos && colon_pos != std::string::npos) {
				fprintf(stderr, "Tried to find Data-Size in wrong data-parse function (txt line %d) [Line %d in function '%s']", line, __LINE__, __func__);
				exit(1); 
			}

			if (dash_pos == std::string::npos && colon_pos == std::string::npos) {
				fprintf(stderr, "Invalid string input (txt line %d) [Line %d in function '%s']", line, __LINE__, __func__);
				exit(1); 
			}

		}

	};

}