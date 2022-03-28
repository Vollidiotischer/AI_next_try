#pragma once


constexpr int num_instances = 5; 

constexpr float window_width = 500; 
constexpr float window_height = 500; 

constexpr float point_size = 5.f; 
constexpr float circle_radius = 10.f; 


struct Point {

	int x, y; 
	char color = 'r'; // 'r' -> red, 'b' -> blue

	Point(int x, int y, char col = 'r') : x(x), y(y), color(col) {}
};