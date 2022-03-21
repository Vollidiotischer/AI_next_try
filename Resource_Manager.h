#pragma once

#include <stdio.h>

#include <vector>
#include <fstream>

#include <string>

namespace SimpleAI {

	struct Resource_Manager {

		static void read_txt_file(std::string path, std::vector<std::string>& result) {

			std::ifstream file; 
			file.open(path); 

			if (file.is_open()) {

				std::string temp_str; 

				while (std::getline(file, temp_str)) {
					result.push_back(temp_str); 
				}

				file.close();
			}
			else {
				fprintf(stderr, "Text File '%s' could not be opened [Line %d in function '%s']", path.c_str(), __LINE__, __func__); 
			}

		}

		static void parse_data(const std::string& inp /* IN */, std::vector<float>& result /* OUT */) {
			
			int last_pos = 0; 

			for (int i = 0; i < inp.size(); i++) {
				if (inp[i] == ':') {
					result.push_back(stof(inp.substr(0, i))); 
					result.push_back(stof(inp.substr(i+1)));
					return; 
				}

				if (inp[i] == '.') {
					result.push_back(stof(inp.substr(last_pos, i - last_pos)));
					last_pos = i + 1; 
				}

				if (inp[i] == '-') {
					result.push_back(stof(inp.substr(last_pos, i - last_pos)));
					result.push_back(stof(inp.substr(i+1))); 
					return; 
				}
			}
		}

	};

}