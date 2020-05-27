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
#include <iostream>
#include <cstdio>
#include <ctime>

#include <vector>
#include <atomic>
#include <thread>

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
	int* searchTree,
	int searchTreeRowSize,
	bool* chosenInTree,
	bool* visitedInCurrentSearch,
	bool* graph,
	int graphSize,
	int* counter);

template<class T>
void swap(int i, int j, T* tab)
{
	T temp = tab[i];
	tab[i] = tab[j];
	tab[j] = temp;
}

template<class T>
void reverse(int i, int j, T* tab)
{
	--j;
	while (i < j)
	{
		swap(i, j, tab);
		++i;
		--j;
	}
}


/// following code is based on std::nextpermutation
/// posible implememtation from cppreference
/// true if the function could rearrange the
/// object as a lexicographicaly greater permutation.
/// Otherwise, the function returns false to indicate
/// that the arrangement is not greater than the previous,
/// but the lowest possible(sorted in ascending order).
template<class T>
bool NextPermutation(int first, int last, T* tab)
{
	if (first == last)
		return false;
	int i = first;
	++i;
	if (i == last)
		return false;
	i = last;
	--i;

	for (;;)
	{
		int ii = i;
		--i;
		if (tab[i] < tab[ii])
		{
			int j = last;
			while (!(tab[i] < tab[--j]))
			{
			}
			swap(i, j, tab);
			reverse(ii, last, tab);
			return true;
		}
		if (i == first)
		{
			reverse(first, last, tab);
			return false;
		}
	}
}

void InitChildSet(
	int root,
	int level,
	int* searchTree,
	int searchTreeRowSize,
	bool* chosenInTree,
	bool* visitedInCurrentSearch,
	bool* graph,
	int graphSize)
{
	searchTree[level * searchTreeRowSize + 0] = 0;
	for (int i = 1; i <= searchTree[(level - 1) *searchTreeRowSize + 0]; ++i)
	{
		if (chosenInTree[searchTree[(level - 1) *searchTreeRowSize + i]])
		{
			int parent = searchTree[(level - 1) * searchTreeRowSize + i];

			for (int a = root + 1; a < graphSize; ++a)
			{
				if (!visitedInCurrentSearch[a] && a != parent)
				{
					if (graph[parent * graphSize + a] || graph[a * graphSize + parent])
					{
						//printf("ADDING %d child of %d \n", a, parent);
						searchTree[level * searchTreeRowSize + ++searchTree[level * searchTreeRowSize + 0]] = a;
						visitedInCurrentSearch[a] = true;
					}
				}
			}
		}
	}
}

void Label2(bool* graph, int n, bool* label)
{
	int vertex_label[MAX_SUBGRAPH_SIZE];
	for (int i = 0; i < n; i++)
		vertex_label[i] = i;

	do
	{
		int result = 0; // 0 - continue, 1 - current is beter, 2 - current is worse
		for (int i = 0; (i < n) && !result; i++)
			for (int j = 0; (j < n) && !result; j++)
			{
				// if current permutation is better
				if (get_edge(vertex_label[i], vertex_label[j], graph) && !get_edge(i, j, label))
					result = 1;
				else if (!get_edge(vertex_label[i], vertex_label[j], graph) && get_edge(i, j, label))
					result = 2;
			}

		// if current is not better
		if (result != 1)
			continue;

		// save current
		for (int i = 0; (i < n); i++)
			for (int j = 0; (j < n); j++)
			{
				bool edge_value = get_edge(vertex_label[i], vertex_label[j], graph);
				set_edge(i, j, label, edge_value);
			}

	} while (NextPermutation(0, n, vertex_label));
}


void Enumerate(
	int root,
	int level,
	int remaining,
	int subgraphSize,
	int* searchTree,
	int searchTreeRowSize,
	bool* chosenInTree,
	bool* visitedInCurrentSearch,
	bool* graph,
	int graphSize,
	std::atomic<int>* counter)
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
				if (graph[vert * graphSize + vert2])
				{
					subgraph[MAX_SUBGRAPH_SIZE * i + j] = true;
				}

			}
		}
		uint largest = 0;
		
		Label(subgraph, subgraphSize, label);
		for (int i = 0; i < MAX_SUBGRAPH_SIZE_SQUARED; i++)
			if (label[i])
				largest += 1 << ((i / MAX_SUBGRAPH_SIZE) * subgraphSize + i % MAX_SUBGRAPH_SIZE);

		++counter[largest];
		return;
	}

	InitChildSet(
		root,
		level,
		searchTree,
		searchTreeRowSize,
		chosenInTree,
		visitedInCurrentSearch,
		graph,
		graphSize);



	for (int k = 1; k <= remaining; ++k)
	{
		if (searchTree[level * searchTreeRowSize + 0] < k)
		{
			return;
		}

		int noNodesOnCurrentLevel = searchTree[level * searchTreeRowSize];
		bool* permutation = new bool[noNodesOnCurrentLevel];
		for (int a = 0; a < noNodesOnCurrentLevel; ++a)
		{
			permutation[a] = false;
		}
		for (int a = 0; a < k; ++a)
		{
			permutation[noNodesOnCurrentLevel - 1 - a] = true;
		}


		// loop over node selection permutations 
		do
		{
			for (int i = 0; i < noNodesOnCurrentLevel; ++i)
			{
				if (permutation[i]) chosenInTree[searchTree[level*searchTreeRowSize +  i + 1]] = true;
			}

			Enumerate(
				root,
				level + 1,
				remaining - k,
				subgraphSize,
				searchTree,
				searchTreeRowSize,
				chosenInTree,
				visitedInCurrentSearch,
				graph,
				graphSize,
				counter);

			for (int a = 0; a < noNodesOnCurrentLevel; ++a)
			{
				chosenInTree[searchTree[level*searchTreeRowSize + a + 1]] = false;
			}
		} while (NextPermutation(0, noNodesOnCurrentLevel, permutation));


		delete permutation;
	}

	for (int i = 1; i <= searchTree[level * searchTreeRowSize + 0]; ++i)
	{
		visitedInCurrentSearch[searchTree[level * searchTreeRowSize + i]] = false;
	}
}

void EnumerateGPU(
	int subgraphSize,
	int* searchTree,
	int searchTreeSize,
	bool* chosenInTree,
	bool* visitedInCurrentSearch,
	bool* graph,
	int graphSize,
	std::atomic<int>* counter,
	int tid,
	int offset
)
{
	if (tid < graphSize)
	{
		int* searchTreeRoot = searchTree + (tid * searchTreeSize * subgraphSize);
		bool* chosenInTreeRoot = chosenInTree + tid * graphSize;
		bool* visitedInCurrentSearchRoot = visitedInCurrentSearch + tid * graphSize;
		searchTreeRoot[0] = 1;
		searchTreeRoot[1] = tid;
		chosenInTreeRoot[tid] = true;
		visitedInCurrentSearchRoot[tid] = true;
			
		Enumerate(
			tid + offset,
			1,
			subgraphSize - 1,
			subgraphSize,
			searchTreeRoot,
			searchTreeSize,
			chosenInTreeRoot,
			visitedInCurrentSearchRoot,
			graph,
			graphSize,
			counter);
	}

}

int GetMaxDeg(bool* graph, int graphSize)
{
	int max = 0;
	for (int i = 0; i < graphSize; ++i)
	{
		int current = 0;
		for (int j = 0; j < graphSize; ++j)
		{
			if (graph[i * graphSize + j]) ++current;
		}

		if (max < current) max = current;
	}
	return max;
}

void ProcessGraphThreading(bool* graph, int graphSize, int* counter, int counterSize, int subgraphSize = SUBGRAPH_SIZE)
{
	const int noBlocksPerRun = 4;
	const int noThreadsPerBlock = 64;
	const int noThreadsPerRun = noBlocksPerRun * noThreadsPerBlock;
	int searchTreeRowSize = GetMaxDeg(graph,graphSize) * (subgraphSize - 2);


	// TODO add errorchecking on allocation
	const size_t searchTreeSize = noThreadsPerRun * subgraphSize * searchTreeRowSize;
	int* searchTree_d = new int[searchTreeSize]();
	for (int i = 0; i < searchTreeSize; i++)
		searchTree_d[i] = 0;

	const size_t chosenInTreeSize = noThreadsPerRun * graphSize;
	bool* chosenInTree_d = new bool[chosenInTreeSize]();
	for (int i = 0; i < chosenInTreeSize; i++)
		chosenInTree_d[i] = 0;

	const size_t visitedInCurrentSearchSize = noThreadsPerRun * graphSize;
	bool* visitedInCurrentSearch_d = new bool[visitedInCurrentSearchSize]();
	for (int i = 0; i < visitedInCurrentSearchSize; i++)
		visitedInCurrentSearch_d[i] = 0;

	const size_t counterSize_d = counterSize;
	std::atomic<int>* counter_d = new std::atomic<int>[counterSize_d]();
	for (int i = 0; i < counterSize_d; i++)
		counter_d[i] = 0;

	const size_t graphSize_d = graphSize * graphSize;
	bool* graph_d = new bool[graphSize_d]();
	for (int i = 0; i < graphSize_d; i++)
		graph_d[i] = graph[i];


	std::thread* threads = new std::thread[noBlocksPerRun * noThreadsPerBlock];

	for (int offset = 0; offset < graphSize; offset += noThreadsPerRun)
	{
		for (int block = 0; block < noBlocksPerRun; block++)
		{
			for (int t = 0; t < noThreadsPerBlock; t++)
			{
				int tid = block * noThreadsPerBlock + t;
				threads[tid] = std::thread(EnumerateGPU,
					subgraphSize,
					searchTree_d,
					searchTreeRowSize,
					chosenInTree_d,
					visitedInCurrentSearch_d,
					graph_d,
					graphSize,
					counter_d,
					tid,
					offset);
			}
			for (int t = 0; t < noThreadsPerBlock; t++)
			{
				int tid = block * noThreadsPerBlock + t;
				threads[tid].join();
			}
		}
	}
	for(int i=0;i< counterSize_d;i++)
		counter[i] = counter_d[i];

	delete [] searchTree_d;
	delete [] chosenInTree_d;
	delete [] visitedInCurrentSearch_d;
	delete [] graph_d;
	delete [] counter_d;

	printf("\n");
}


////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
	srand(time(NULL));

	int* counter = new int[SUBGRAPH_INDEX_SIZE]();
	int* count_master = new int[SUBGRAPH_INDEX_SIZE]();
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

	// TODO change loader so that it returns one dim array
	bool* graph_one_dim = new bool[graphSize * graphSize]();
	for (int i = 0; i < graphSize; ++i)
	{
		for (int j = 0; j < graphSize; ++j)
		{
			graph_one_dim[i * graphSize + j] = graph[i][j];
		}
	}
	std::cout << std::endl << "Processing graph" << std::endl;

	std::clock_t start;
	double duration;

	start = std::clock();
	ProcessGraphThreading(graph_one_dim, graphSize, count_master, SUBGRAPH_INDEX_SIZE);
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

	std::cout << "ProcessGraphThreading duration: " << duration << '\n';
	std::cout << "graphID\tcount" << std::endl;
	for (uint i = 0; i < SUBGRAPH_INDEX_SIZE; ++i)
	{
		if (count_master[i])
		{
			std::cout << i << "\t" << count_master[i] << std::endl;
		}
	}

	std::cout << std::endl << "Generating random graphs" << std::endl;
	for (int i = 0; i < RANDOM_GRAPH_NUMBER; i++)
	{
		std::cout << ".";
		//std::cout << std::endl << "Generating random graph " << i << std::endl;
		GenerateGraph(graph, graphSize);
		for (int i = 0; i < graphSize; ++i)
		{
			for (int j = 0; j < graphSize; ++j)
			{
				graph_one_dim[i * graphSize + j] = graph[i][j];
			}
		}
		//std::cout << std::endl << "Processing graph" << std::endl;
//		ProcessGraph(graph_one_dim, graphSize, counter);

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
		}
	}

	delete graph_one_dim;

	return 0;
}

