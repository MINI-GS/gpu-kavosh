#include "random_graph.h"

#include <tuple>
#include <iostream>

std::tuple<int, int> GetRandomEdge(bool** graph, int n, int edge_count)
{
	int random_edge_number = rand() % edge_count;
	int edge_number = -1;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			edge_number += graph[i][j];
			if (edge_number == random_edge_number)
				return std::make_tuple(i, j);
		}
	}
	std::cout << edge_count << ":::" << edge_number << std::endl;
	throw "Something weird happened - GetRandomEdge";
}


void RemoveEdge(bool** graph, int a, int b)
{
	if (!graph[a][b])
		throw "xx";
	graph[a][b] = 0;
}


void AddEdge(bool** graph, int a, int b)
{
	if (graph[a][b])
		throw "xx";
	graph[a][b] = 1;
}


bool IsConnected(bool** graph, int a, int b)
{
	return graph[a][b];
}


void GenerateGraph(bool** graph, int n)
{
	int edge_count = 0;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			edge_count += graph[i][j];
	}

	int numOfExchange = 10 * edge_count;

	for (int i = 0; i < numOfExchange; i++)
	{
		int a, b, c, d;
		do
		{
			std::tie(a, c) = GetRandomEdge(graph, n, edge_count);
			std::tie(b, d) = GetRandomEdge(graph, n, edge_count);
		} while (
			!IsConnected(graph, a, c) ||
			!IsConnected(graph, b, d) ||
			IsConnected(graph, a, d) ||
			IsConnected(graph, b, c) ||
			c == b ||
			a == d ||
			a == b ||
			c == d
			);

		RemoveEdge(graph, a, c);
		RemoveEdge(graph, b, d);
		AddEdge(graph, a, d);
		AddEdge(graph, b, c);
	}
}
