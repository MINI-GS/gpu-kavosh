#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <assert.h>
#include <random>
#include <float.h>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "config.h"
#include "labeling.cuh"
#include "loader.h"


#define uint unsigned int
#define ull unsigned long long

//#define DEBUG

__host__ __device__ void Enumerate(
	int root,
	int level,
	int remaining,
	int subgraphSize,
	int** searchTree,
	bool* chosenInTree,
	bool* visitedInCurrentSearch,
	bool** graph,
	int graphSize,
	int* counter);

__host__ __device__ void RevolveR(
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
	int* counter);

/*
__host__ __device__ void Vertical(
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



__host__ __device__ void Revolve(
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
	int* counter
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

__host__ __device__ void RevolveR(
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
	int* counter)
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

__host__ __device__ void InitChildSet(
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

__host__ __device__ void Enumerate(
	int root,
	int level,
	int remaining,
	int subgraphSize,
	int** searchTree,
	bool* chosenInTree,
	bool* visitedInCurrentSearch,
	bool** graph,
	int graphSize,
	int* counter)
{
	if (remaining == 0)
	{
		int arrH[MAX_SUBGRAPH_SIZE];
		int arrV[MAX_SUBGRAPH_SIZE];

		for (int i = 0; i < 8; ++i)
		{
			arrH[i] = arrV[i] = i;
		}
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
				largest += 1 << ((i/ MAX_SUBGRAPH_SIZE)* SUBGRAPH_SIZE+i% MAX_SUBGRAPH_SIZE);
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


	int graphSize = -1;
	bool** graph = Load(&graphSize, "data/allActors.csv", "data/allActorsRelation.csv");
	/*bool** graph = new bool* [7];

	for (int i = 0; i < 7; ++i)
	{
		graph[i] = new bool[7];
		for (int j = 0; j < 7; j++)
			graph[i][j] = 0;
	}

	graph[0][1] = true;
	graph[0][2] = true;
	graph[1][5] = true;
	graph[2][6] = true;
	graph[2][4] = true;
	graph[2][5] = true;
	graph[3][2] = true;
	graph[4][3] = true;
	graph[4][5] = true;
	graph[4][2] = true;
	graph[4][0] = true;
	graph[5][6] = true;
	graph[5][3] = true;
	graph[6][1] = true;

	int graphSize = 7;*/
	int counter[131071];
	for (int i = 0; i < 131071; ++i)
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

	for (uint i = 0; i < 131071; ++i)
	{
		if (counter[i] != 0)
		{
			printf("\n%d %d", i, counter[i]);
			for (int a = 0; a < 16; ++a)
			{
				if (a % 4 == 0) printf("\n");
				printf("%d", (i & (1 << (15 - a))) == 0 ? 0 : 1);
			}
		}
	}

	printf("\nHELLO");
}

