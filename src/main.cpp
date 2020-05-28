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
#include <sstream>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <device_atomic_functions.h>

#include "config.h"
#include "labeling.cuh"
#include "loader.h"
#include "random_graph.h"
#include "enumeration_cpu_multi.h"
#include "enumeration_cpu_single.h"

#include <map>


////////////////////////////////////////////////////////////////////////////////
// GPU


#define uint unsigned int
#define ull unsigned long long

//#define DEBUG

__device__ int label_type = 1;


__host__ __device__ void InitChildSet(
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
	for (int i = 1; i <= searchTree[(level - 1) * searchTreeRowSize + 0]; ++i)
	{
		if (chosenInTree[searchTree[(level - 1) * searchTreeRowSize + i]])
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




__device__ void Enumerate(
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
	int* counter)
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

		if (label_type == 2)
		{
			Label2(subgraph, subgraphSize, label);
		}
		else
		{
			Label3(subgraph, subgraphSize, label);
		}

		for (int i = 0; i < MAX_SUBGRAPH_SIZE_SQUARED; i++)
			if (label[i])
				largest += 1 << ((i / MAX_SUBGRAPH_SIZE) * subgraphSize + i % MAX_SUBGRAPH_SIZE);

		atomicAdd(counter + largest, 1);
		//++counter[largest];
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
			break;
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
				if (permutation[i]) chosenInTree[searchTree[level * searchTreeRowSize + i + 1]] = true;
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
				chosenInTree[searchTree[level * searchTreeRowSize + a + 1]] = false;
			}
		} while (NextPermutation(0, noNodesOnCurrentLevel, permutation));


		delete permutation;
	}

	for (int i = 1; i <= searchTree[level * searchTreeRowSize + 0]; ++i)
	{
		visitedInCurrentSearch[searchTree[level * searchTreeRowSize + i]] = false;
	}
}

__global__ void EnumerateGPU(
	int subgraphSize,
	int* searchTree,
	int searchTreeSize,
	bool* chosenInTree,
	bool* visitedInCurrentSearch,
	bool* graph,
	int graphSize,
	int* counter,
	int offset
)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

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

void ProcessGraphGPU(bool* graph, int graphSize, int* counter, int counterSize, int subgraphSize = SUBGRAPH_SIZE)
{
	const int noBlocksPerRun = 4;
	const int noThreadsPerBlock = 64;
	const int noThreadsPerRun = noBlocksPerRun * noThreadsPerBlock;
	int searchTreeRowSize = GetMaxDeg(graph, graphSize) * (subgraphSize - 2);

	// TODO add errorchecking on allocation

	int* searchTree_d;
	const size_t searchTreeSize = noThreadsPerRun * subgraphSize * searchTreeRowSize * sizeof(int);
	printf("Allocating %f GB for search tree array\n", (double)searchTreeSize / BYTES_IN_GIGABYTE);
	cudaMalloc((void**)&searchTree_d, searchTreeSize);
	cudaMemset(searchTree_d, 0, searchTreeSize);

	bool* chosenInTree_d;
	const size_t chosenInTreeSize = noThreadsPerRun * graphSize * sizeof(bool);
	printf("Allocating %f GB for chosen in tree array\n", (double)chosenInTreeSize / BYTES_IN_GIGABYTE);
	cudaMalloc((void**)&chosenInTree_d, chosenInTreeSize);
	cudaMemset(chosenInTree_d, 0, chosenInTreeSize);

	bool* visitedInCurrentSearch_d;
	const size_t visitedInCurrentSearchSize = noThreadsPerRun * graphSize * sizeof(bool);
	printf("Allocating %f GB for visited in current seatch array\n", (double)visitedInCurrentSearchSize / BYTES_IN_GIGABYTE);
	cudaMalloc((void**)&visitedInCurrentSearch_d, visitedInCurrentSearchSize); // one thread gets its own list (len graph size)
	cudaMemset(visitedInCurrentSearch_d, 0, visitedInCurrentSearchSize);


	int* counter_d;
	const size_t counterSize_d = counterSize * sizeof(int);
	printf("Allocating %f GB for counter array\n", (double)counterSize_d / BYTES_IN_GIGABYTE);
	cudaMalloc((void**)&counter_d, counterSize_d); // counter is common for all
	cudaMemset(counter_d, 0, counterSize_d);

	bool* graph_d;
	const size_t graphSize_d = graphSize * graphSize * sizeof(bool);
	printf("Allocating %f GB for graph array\n", (double)graphSize_d / BYTES_IN_GIGABYTE);
	cudaMalloc((void**)&graph_d, graphSize_d); // graph is common
	cudaMemcpy(graph_d, graph, graphSize_d, cudaMemcpyHostToDevice);


	// TODO start on more threads (one should add more memory)
	//     and process all vertices (as a root)
	printf("Lauching kernel\n");
	EnumerateGPU << <noBlocksPerRun, noThreadsPerBlock >> > (
		subgraphSize,
		searchTree_d,
		searchTreeRowSize,
		chosenInTree_d,
		visitedInCurrentSearch_d,
		graph_d,
		graphSize,
		counter_d,
		0);
	cudaDeviceSynchronize();
	cudaError_t code = cudaGetLastError();

	if (code != cudaSuccess)
	{
		fprintf(stderr, "kernellAssert: %s\n", cudaGetErrorString(code));
		if (abort) exit(code);
	}

	printf("Copying couter to host\n");
	cudaMemcpy(counter, counter_d, counterSize * sizeof(int), cudaMemcpyDeviceToHost);

	printf("Device memory deallocation\n");
	cudaFree(searchTree_d);
	cudaFree(chosenInTree_d);
	cudaFree(visitedInCurrentSearch_d);
	cudaFree(graph_d);
	cudaFree(counter_d);

	printf("\n");
}

////////////////////////////////////////////////////////////////////////////////
// INPUT

int ParseArgsFromConsole(int& enumeration_strategy, int& labeling_strategy, int& max_input_graph_id, int& number_of_generated_graphs)
{
	std::cout << "Choose enumeration strategy" << std::endl <<
		"1 - GPU / 2 - CPU one thread / 3 - CPU multithreading" << std::endl;
	std::cin >> enumeration_strategy;

	std::cout << "Choose labeling strategy" << std::endl;
	if (enumeration_strategy != 1)
		std::cout << "1 - Heap's algorithm recurrent / ";
	std::cout << "2 - algorithm based on std::nextpermutation / 3 - Heap's algorithm non recurrent" << std::endl;
	std::cin >> labeling_strategy;

	cudaMemcpyToSymbol(label_type, &labeling_strategy, sizeof(int), 0, cudaMemcpyHostToDevice);

	std::cout << "How much do you want to reduce the input graph? Pass nuber equal to max allowed vertex number from input graph. If want to use unreduced graph pass -1. " << std::endl;
	std::cin >> max_input_graph_id;

	std::cout << "How many random graphs do you want to generate?" << std::endl;
	std::cin >> number_of_generated_graphs;

	return 0;
}

int ParseInt(char* c_s, int& result)
{
	std::istringstream ss(c_s);
	int tmp;
	if (!(ss >> tmp)) {
		std::cerr << "Invalid number: " << c_s[1] << '\n';
		return 1;
	}
	else if (!ss.eof()) {
		std::cerr << "Trailing characters after number: " << c_s[1] << '\n';
		return 1;
	}
	result = tmp;
	return 0;
}

int ParseArgs(int argc, char** argv,
	int& enumeration_strategy, int& labeling_strategy, int& max_input_graph_id, int& number_of_generated_graphs)
{
	if (ParseInt(argv[1], enumeration_strategy))
		return 1;
	if (ParseInt(argv[2], labeling_strategy))
		return 1;
	if (ParseInt(argv[3], max_input_graph_id))
		return 1;
	if (ParseInt(argv[4], number_of_generated_graphs))
		return 1;
	return 0;
}

int ValidateArgs(int enumeration_strategy, int labeling_strategy)
{
	if (enumeration_strategy < 1 || enumeration_strategy>3)
	{
		std::cout << "Bad input" << std::endl;
		return 1;
	}
	if (labeling_strategy < 1 || labeling_strategy>3 || (enumeration_strategy == 1 && labeling_strategy == 1))
	{
		std::cout << "Bad input" << std::endl;
		return 1;
	}
	return 0;
}

////////////////////////////////////////////////////////////////////////////////
// MAIN

int main(int argc, char** argv)
{
	srand(time(NULL));

	int* counter = new int[SUBGRAPH_INDEX_SIZE];
	int* count_master = new int[SUBGRAPH_INDEX_SIZE];
	std::map <int, double> mean;
	std::map <int, double> var;
	std::map <int, double> score;

	int enumeration_strategy = 0;
	int labeling_strategy = 0;
	int max_input_graph_id = -1;
	int number_of_generated_graphs = 0;

	if (argc == 5)
	{
		if (ParseArgs(argc, argv, enumeration_strategy, labeling_strategy, max_input_graph_id, number_of_generated_graphs))
			return 1;
	}
	else
	{
		if (ParseArgsFromConsole(enumeration_strategy, labeling_strategy, max_input_graph_id, number_of_generated_graphs))
			return 1;
	}
	if (ValidateArgs(enumeration_strategy, labeling_strategy))
		return 1;

	std::cout << std::endl << "Loading graph from files data/allActors.csv (vertices) and data/allActorsRelation.csv (edges)" << std::endl;
	int graphSize = -1;
	bool** graph = Load(&graphSize, "data/allActors.csv", "data/allActorsRelation.csv", max_input_graph_id);

	// TODO change loader so that it returns one dim array
	bool* graph_one_dim = new bool[graphSize * graphSize];
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
	switch (enumeration_strategy)
	{
	case 1:
		ProcessGraphGPU(graph_one_dim, graphSize, count_master, SUBGRAPH_INDEX_SIZE, SUBGRAPH_SIZE);
		break;
	case 2:
		EnumerationSingle::ProcessGraph(graph_one_dim, graphSize, count_master, SUBGRAPH_INDEX_SIZE, SUBGRAPH_SIZE, labeling_strategy);
		break;
	case 3:
		EnumerationMulti::ProcessGraph(graph_one_dim, graphSize, count_master, SUBGRAPH_INDEX_SIZE, SUBGRAPH_SIZE, labeling_strategy);
		break;
	default:
		return 1;
	}
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

	std::cout << "ProcessGraph duration: " << duration << std::endl << std::endl;
	std::cout << "graphID\tcount" << std::endl;
	for (uint i = 0; i < SUBGRAPH_INDEX_SIZE; ++i)
	{
		if (count_master[i])
		{
			std::cout << i << "\t" << count_master[i] << std::endl;
		}
	}

	std::cout << std::endl << "Generating random graphs" << std::endl;
	for (int i = 0; i < number_of_generated_graphs; i++)
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
		switch (enumeration_strategy)
		{
		case 1:
			ProcessGraphGPU(graph_one_dim, graphSize, counter, SUBGRAPH_INDEX_SIZE, SUBGRAPH_SIZE);
			break;
		case 2:
			EnumerationSingle::ProcessGraph(graph_one_dim, graphSize, counter, SUBGRAPH_INDEX_SIZE, SUBGRAPH_SIZE, labeling_strategy);
			break;
		case 3:
			EnumerationMulti::ProcessGraph(graph_one_dim, graphSize, counter, SUBGRAPH_INDEX_SIZE, SUBGRAPH_SIZE, labeling_strategy);
			break;
		default:
			return 1;
		}

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
		mean[i] = mean[i] / (double)number_of_generated_graphs;
		var[i] = sqrt((var[i] - ((double)number_of_generated_graphs * (mean[i] * mean[i]))) / (double)number_of_generated_graphs);

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

