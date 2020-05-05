#include "loader.h"

#include <fstream>
#include <map>
#include <vector>
#include <iostream>


bool** Load(int* n, std::string filename_v, std::string filename_e)
{
	std::fstream file_v;
	std::fstream file_e;

	file_v.open(filename_v, std::ios::in);
	file_e.open(filename_e, std::ios::in);

	std::map<int, int> ids;

	std::string line;
	int vertice_count = 0;
	std::getline(file_v, line);  // skip labels
	while (std::getline(file_v, line))
	{
		//std::cout << line << std::endl;

		std::string code = line.substr(line.find_last_of(";") + 1);
		//std::cout << code << std::endl;

		int id = std::stoi(code);
		ids[id] = vertice_count;
		vertice_count++;
	}

	bool** graph = new bool* [vertice_count];
	for (int i = 0; i < vertice_count; ++i)
	{
		graph[i] = new bool[vertice_count];
		for (int j = 0; j < vertice_count; j++)
			graph[i][j] = 0;
	}

	std::getline(file_e, line);  // skip labels
	while (std::getline(file_e, line))
	{
		//std::cout << line << std::endl;

		std::string code = line.substr(0, line.find(";"));
		int id1 = std::stoi(code);
		id1 = ids[id1];
		line = line.substr(line.find(";")+1);
		code = line.substr(0, line.find(";"));
		int id2 = std::stoi(code);
		id2 = ids[id2];

		graph[id1][id2] = true;
		graph[id2][id1] = true;
	}

	return graph;
}
