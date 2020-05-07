#include "random_graph.h"

#include <tuple>
#include <iostream>
#include <chrono> 
using namespace std::chrono;


/*
3 sposoby na generowanie losowych krawêdzi
1 i 2 - takie same prawdopodobieñstwo na wszystkie krawêdzie
3 - takie same prawdopodobieñstwo na wszystkie wierzcho³ki
1 i 2 mniej wiêcej ten sam czas wykonania
3 jest 20 razy szybszy ni¿ 1 i 2 dla naszego grafu
W kavoshu jest wybrany sposób 3 (chocia¿ s¹ ró¿nice w algorytmach ju¿ po wybraniu krawêdzi,
i sam sposób wyboru jest te¿ trochê zoptymalizowany)
*/

std::tuple<int, int> GetRandomEdge1(bool** graph, int n, int edge_count)
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

std::tuple<int, int> GetRandomEdge2(bool** graph, int n, int edge_count)
{
	return std::make_tuple(rand() % n, rand() % n);
}

std::tuple<int, int> GetRandomEdge3(bool** graph, int n, int edge_count)
{
	int a;
	do
	{
		a = rand() % n;
		edge_count = 0;
		for (int j = 0; j < n; j++)
			edge_count += graph[a][j];
	} while (edge_count == 0);

	int random_edge_number = rand() % edge_count;
	int edge_number = -1;
	for (int j = 0; j < n; j++)
	{
		edge_number += graph[a][j];
		if (edge_number == random_edge_number)
			return std::make_tuple(a, j);
	}
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
	auto start = high_resolution_clock::now();

	int edge_count = 0;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			edge_count += graph[i][j];
	}

	int numOfExchange = 10 * edge_count;

	long long generated_edges = 0;
	for (int i = 0; i < numOfExchange; i++)
	{
		int a, b, c, d;
		do
		{
			std::tie(a, c) = GetRandomEdge3(graph, n, edge_count);
			std::tie(b, d) = GetRandomEdge3(graph, n, edge_count);
			generated_edges++;
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

	auto stop = high_resolution_clock::now();

	auto duration = duration_cast<microseconds>(stop - start);

	std::cout << "Time taken by graph generation: "
		<< duration.count() << " microseconds" << std::endl;
	std::cout << "Avarage generated edge pairs: " << generated_edges / (numOfExchange * 1.0) << std::endl;
}
