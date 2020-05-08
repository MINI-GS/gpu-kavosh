#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <assert.h>
#include <random>
#include <time.h>
#include <float.h>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "config.h"
#include "labeling.cuh"
#include "loader.h"
#include "random_graph.h"

#include <map>


#define uint unsigned int
#define ull unsigned long long

//#define DEBUG

void Enumerate(
	int root,
	int level,
	int remaining,
	int subgraphSize,
	int** searchTree,
	bool* chosenInTree,
	bool* visitedInCurrentSearch,
	bool** graph,
	int graphSize,
	std::map<int, int>& counter);

void RevolveR(
	int n,
	int left,
	int right,

	int root,
	int level,
	int remaining,
	int subgraphSize,
	int** searchTree,
	bool* chosenInTree,
	bool* visitedInCurrentSearch,
	bool** graph,
	int graphSize,
	std::map<int, int>& counter);

/*
 void Vertical(
		int* arrH,
		int* arrV,
		int arrLen,
		int level
		)
{
	if(level == arrLen)
	{
		for(int i = 0; i < arrLen; ++i)
		{
			printf("%d,", arrH[i]);
		}
		printf("\n");
		for(int i = 0; i < arrLen; ++i)
		{
			printf("%d,", arrV[i]);
		}

		printf("\n");
		printf("\n");
		return;
	}
	for(int i = level; i < arrLen; ++i)
	{
		int tmp = arrV[level];
		arrV[level] = arrV[i];
		arrV[i] = tmp;
		Vertical(arrH, arrV, arrLen, level + 1);
		tmp = arrV[level];
		arrV[level] = arrV[i];
		arrV[i] = tmp;
	}
}*/



void Revolve(
	int n,
	int left,
	int right,

	int root,
	int level,
	int remaining,
	int subgraphSize,
	int** searchTree,
	bool* chosenInTree,
	bool* visitedInCurrentSearch,
	bool** graph,
	int graphSize,
	std::map<int, int>& counter
)
{
	int* tab = searchTree[level];

	if (n == 0)
	{
		for (int i = left; i < right; ++i)
		{
			chosenInTree[tab[i]] = false;
		}

		Enumerate(
			root,
			level + 1,
			remaining,
			subgraphSize,
			searchTree,
			chosenInTree,
			visitedInCurrentSearch,
			graph,
			graphSize,
			counter);

	}
	else if (n == right - left)
	{
		for (int i = left; i < right; ++i)
		{
			chosenInTree[tab[i]] = true;
		}

		Enumerate(
			root,
			level + 1,
			remaining,
			subgraphSize,
			searchTree,
			chosenInTree,
			visitedInCurrentSearch,
			graph,
			graphSize,
			counter);

		for (int i = left; i < right; ++i)
		{
			chosenInTree[tab[i]] = false;
		}
	}
	else
	{
		chosenInTree[tab[left]] = false;
		Revolve(n, left + 1, right,
			root,
			level,
			remaining,
			subgraphSize,
			searchTree,
			chosenInTree,
			visitedInCurrentSearch,
			graph,
			graphSize,
			counter);

		chosenInTree[tab[left]] = true;
		RevolveR(n - 1, left + 1, right,
			root,
			level,
			remaining,
			subgraphSize,
			searchTree,
			chosenInTree,
			visitedInCurrentSearch,
			graph,
			graphSize,
			counter);


		chosenInTree[tab[left]] = false;
	}
}

void RevolveR(
	int n,
	int left,
	int right,

	int root,
	int level,
	int remaining,
	int subgraphSize,
	int** searchTree,
	bool* chosenInTree,
	bool* visitedInCurrentSearch,
	bool** graph,
	int graphSize,
	std::map<int, int>& counter)
{
	int* tab = searchTree[level];

	if (n == 0)
	{
		for (int i = left; i < right; ++i)
		{
			chosenInTree[tab[i]] = false;
		}

		Enumerate(
			root,
			level + 1,
			remaining,
			subgraphSize,
			searchTree,
			chosenInTree,
			visitedInCurrentSearch,
			graph,
			graphSize,
			counter);
	}
	else if (n == right - left)
	{
		for (int i = left; i < right; ++i)
		{
			chosenInTree[tab[i]] = true;
		}

		Enumerate(
			root,
			level + 1,
			remaining,
			subgraphSize,
			searchTree,
			chosenInTree,
			visitedInCurrentSearch,
			graph,
			graphSize,
			counter);

		for (int i = left; i < right; ++i)
		{
			chosenInTree[tab[i]] = false;
		}
	}
	else
	{
		chosenInTree[tab[right - 1]] = false;
		RevolveR(n, left, right - 1,
			root,
			level,
			remaining,
			subgraphSize,
			searchTree,
			chosenInTree,
			visitedInCurrentSearch,
			graph,
			graphSize,
			counter);

		chosenInTree[tab[right - 1]] = true;
		Revolve(n - 1, left, right - 1,
			root,
			level,
			remaining,
			subgraphSize,
			searchTree,
			chosenInTree,
			visitedInCurrentSearch,
			graph,
			graphSize,
			counter);


		chosenInTree[tab[right - 1]] = false;
	}
}

void InitChildSet(
	int root,
	int level,
	int** searchTree,
	bool* chosenInTree,
	bool* visitedInCurrentSearch,
	bool** graph,
	int graphSize)
{
	searchTree[level][0] = 0;
	for (int i = 1; i <= searchTree[level - 1][0]; ++i)
	{
		if (chosenInTree[searchTree[level - 1][i]])
		{
			int parent = searchTree[level - 1][i];

			for (int a = root + 1; a < graphSize; ++a)
			{
				if (!visitedInCurrentSearch[a] && a != parent)
				{
					if (graph[parent][a] || graph[a][parent])
					{
						//printf("ADDING %d child of %d \n", a, parent);
						searchTree[level][++searchTree[level][0]] = a;
						visitedInCurrentSearch[a] = true;
					}
				}
			}
		}
	}
}

void Enumerate(
	int root,
	int level,
	int remaining,
	int subgraphSize,
	int** searchTree,
	bool* chosenInTree,
	bool* visitedInCurrentSearch,
	bool** graph,
	int graphSize,
	std::map<int, int>& counter)
{
	if (remaining == 0)
	{
		bool subgraph[MAX_SUBGRAPH_SIZE_SQUARED];
		for (int i = 0; i < MAX_SUBGRAPH_SIZE_SQUARED; ++i) subgraph[i] = false;
		bool label[MAX_SUBGRAPH_SIZE_SQUARED];
		for (int i = 0; i < MAX_SUBGRAPH_SIZE_SQUARED; ++i) label[i] = false;

		int chosenVerts[MAX_SUBGRAPH_SIZE];
		int iter = 0;
		for (int i = 0; i < graphSize; ++i)
		{
			if (chosenInTree[i]) chosenVerts[iter++] = i;
		}

		for (int i = 0; i < subgraphSize; ++i)
		{
			int vert = chosenVerts[i];
			for (int j = 0; j < subgraphSize; ++j)
			{
				int vert2 = chosenVerts[j];
				if (vert2 == vert) continue;
				if (graph[vert][vert2])
				{
					subgraph[MAX_SUBGRAPH_SIZE * i + j] = true;
				}

			}
		}
		uint largest = 0;
		Label(subgraph, SUBGRAPH_SIZE, label);
		for (int i = 0; i < MAX_SUBGRAPH_SIZE_SQUARED; i++)
			if (label[i])
				largest += 1 << ((i / MAX_SUBGRAPH_SIZE) * SUBGRAPH_SIZE + i % MAX_SUBGRAPH_SIZE);
		//Horisontal(arrH, arrV, subgraphSize, 0, subgraph, &largest);
		++counter[largest];

#ifdef DEBUG

#ifdef LEVELS
		for (int lvl = 0; lvl < level; ++lvl)
		{
			printf("%d LEVEL %d:\t", searchTree[lvl][0], lvl);
			for (int i = 1; i <= searchTree[lvl][0]; ++i)
			{
				printf("%d", searchTree[lvl][i] + 1);
			}
			printf("\n");
		}
#endif
		printf("SUBGRAP:\t");
		for (int i = 0; i < graphSize; ++i)
		{
			if (chosenInTree[i]) printf("%d", i + 1);
		}
		printf("\n");
#endif
		return;
	}

	InitChildSet(
		root,
		level,
		searchTree,
		chosenInTree,
		visitedInCurrentSearch,
		graph,
		graphSize);


	for (int k = 1; k <= remaining; ++k)
	{
		if (searchTree[level][0] < k)
		{
			return;
		}

		Revolve(
			k,
			1,
			searchTree[level][0] + 1,
			root,
			level,
			remaining - k,
			subgraphSize,
			searchTree,
			chosenInTree,
			visitedInCurrentSearch,
			graph,
			graphSize,
			counter);
	}

	for (int i = 1; i <= searchTree[level][0]; ++i)
	{
		visitedInCurrentSearch[searchTree[level][i]] = false;
	}
}


// zmieniæ na int* i zwracaæ counter?
void ProcessGraph(bool** graph, int graphSize, std::map<int, int>& counter)
{
	int root = 0;
	int level = 1;
	int remaring = 3;
	int subgraphSize = SUBGRAPH_SIZE;
	int** searchTree = new int* [5];

	for (int i = 0; i < 5; ++i)
	{
		searchTree[i] = new int[SEARCH_TREE_SIZE];
		for (int j = 0; j < SEARCH_TREE_SIZE; j++)
			searchTree[i][j] = 0;
	}

	searchTree[0][0] = 1;
	searchTree[0][1] = root;

	bool* chosenInTree = new bool[SEARCH_TREE_SIZE];
	for (int i = 0; i < SEARCH_TREE_SIZE; i++)
		chosenInTree[i] = 0;
	chosenInTree[root] = true;
	bool* visitedInCurrentSearch = new bool[SEARCH_TREE_SIZE];
	for (int i = 0; i < SEARCH_TREE_SIZE; i++)
		visitedInCurrentSearch[i] = 0;


	for (int i = 0; i < SUBGRAPH_INDEX_SIZE; ++i)
	{
		counter[i] = 0;
	}
	Enumerate(
		root,
		level,
		remaring,
		subgraphSize,
		searchTree,
		chosenInTree,
		visitedInCurrentSearch,
		graph,
		graphSize,
		counter);





}


////////////////////////////////////////////////////////////////////////////////
/*
 * 		int root,
		int level,
		int remaining,
		int subgraphSize,
		int** searchTree,
		bool* chosenInTree,
		bool* visitedInCurrentSearch,
		bool** graph,
		int graphSize)
 */
int main(int argc, char** argv)
{
	srand(time(NULL));

	std::cout << SUBGRAPH_INDEX_SIZE << std::endl;
	std::map <int, int> counter;
	std::map <int, int> count_master;
	std::map <int, double> mean;
	std::map <int, double> var;
	std::map <int, double> score;

	/*for (int i = 0; i < SUBGRAPH_INDEX_SIZE; i++)
	{
		counter[i]=0;
		count_master[i]=0;
		mean[i]=0;
		var[i]=0;
	}*/


	std::cout << std::endl << "Loading graph from file" << std::endl;
	int graphSize = -1;
	bool** graph = Load(&graphSize, "data/allActors.csv", "data/allActorsRelation.csv");

	std::cout << std::endl << "Processing graph" << std::endl;
	ProcessGraph(graph, graphSize, count_master);

	std::cout << "graphID\tcount" << std::endl;
	for (uint i = 0; i < SUBGRAPH_INDEX_SIZE; ++i)
	{
		if (count_master[i])
		{
			std::cout << i << "\t" << count_master[i] << std::endl;
			/*for (int a = 0; a < 16; ++a)
			{
				if (a % 4 == 0) printf("\n");
				printf("%d", (i & (1 << (15 - a))) == 0 ? 0 : 1);
			}*/
		}
	}

	std::cout << std::endl << "Generating random graphs" << std::endl;
	for (int i = 0; i < RANDOM_GRAPH_NUMBER; i++)
	{
		std::cout << ".";
		//std::cout << std::endl << "Generating random graph " << i << std::endl;
		GenerateGraph(graph, graphSize);

		//std::cout << std::endl << "Processing graph" << std::endl;
		ProcessGraph(graph, graphSize, counter);

		for (int j = 0; j < SUBGRAPH_INDEX_SIZE; j++)
		{
			mean[j] += counter[j];

			var[j] += counter[j] * counter[j];

		}
	}
	std::cout << std::endl;

	std::cout << std::endl << "Calculating score" << std::endl;

	for (int i = 0; i < SUBGRAPH_INDEX_SIZE; i++)
	{
		mean[i] = mean[i] / (double)RANDOM_GRAPH_NUMBER;
		var[i] = sqrt((var[i] - ((double)RANDOM_GRAPH_NUMBER * (mean[i] * mean[i]))) / (double)RANDOM_GRAPH_NUMBER);

		if (var[i] != 0)
			score[i] = (count_master[i] - mean[i]) / var[i];
		else
			score[i] = -1;

	}

	std::cout << "graphID\tscore" << std::endl;
	for (uint i = 0; i < SUBGRAPH_INDEX_SIZE; ++i)
	{
		if (count_master[i])
		{
			std::cout << i << "\t" << score[i] << std::endl;
			/*for (int a = 0; a < 16; ++a)
			{
				if (a % 4 == 0) printf("\n");
				printf("%d", (i & (1 << (15 - a))) == 0 ? 0 : 1);
			}*/
		}
	}

	return 0;
}

