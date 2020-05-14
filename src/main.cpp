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
#include <device_atomic_functions.h>

#include "config.h"
#include "labeling.cuh"
#include "loader.h"
#include "random_graph.h"

#include <map>


#define uint unsigned int
#define ull unsigned long long

//#define DEBUG

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
	int* counter);

template<class T>
__host__ __device__ void swap(int i, int j, T* tab)
{
	int temp = tab[i];
	tab[i] = tab[j];
	tab[j] = temp;
}

template<class T>
__host__ __device__ void reverse(int i, int j, T* tab)
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
__host__ __device__ bool NextPermutation(int first, int last, T* tab)
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
		// TODO make Label less recursive
		Label(subgraph, subgraphSize, label);
		for (int i = 0; i < MAX_SUBGRAPH_SIZE_SQUARED; i++)
			if (label[i])
				largest += 1 << ((i / MAX_SUBGRAPH_SIZE) * subgraphSize + i % MAX_SUBGRAPH_SIZE);

		atomicAdd(counter + largest, 1);

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

__global__ void EnumerateGPU(
	int subgraphSize,
	int* searchTree,
	int searchTreeSize,
	bool* chosenInTree,
	bool* visitedInCurrentSearch,
	bool* graph,
	int graphSize,
	int* counter
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
			tid,
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

void ProcessGraphGPU(bool* graph, int graphSize, int* counter, int counterSize, int subgraphSize = 3)
{
	int root = 0;
	int level = 1;
	int remaring = 3;

	const int noBlocksPerRun = 2;
	const int noThreadsPerBlock = 256;
	const int noThreadsPerRun = noBlocksPerRun * noThreadsPerBlock;
	int searchTreeRowSize = GetMaxDeg(graph,graphSize) * (subgraphSize - 2);

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
	EnumerateGPU<<<noBlocksPerRun,noThreadsPerBlock>>>(
		subgraphSize,
		searchTree_d,
		searchTreeRowSize,
		chosenInTree_d,
		visitedInCurrentSearch_d,
		graph_d,
		graphSize,
		counter_d);
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


// zmieniæ na int* i zwracaæ counter?
//void ProcessGraph(bool* graph, int graphSize, int* counter)
//{
//	int root = 0;
//	int level = 1;
//	int subgraphSize = SUBGRAPH_SIZE;
//	int remaring = subgraphSize - 1;
//	int* searchTree = new int [5 * SEARCH_TREE_SIZE];
//	bool* chosenInTree = new bool[SEARCH_TREE_SIZE];
//	bool* visitedInCurrentSearch = new bool[SEARCH_TREE_SIZE];
//	int searchTreeRowSize = SEARCH_TREE_SIZE;
//	for (int i = 0; i < 5; ++i)
//	{
//		for (int j = 0; j < SEARCH_TREE_SIZE; j++)
//			searchTree[i * searchTreeRowSize + j] = 0;
//	}
//
//	for (int i = 0; i < SUBGRAPH_INDEX_SIZE; ++i)
//	{
//		counter[i] = 0;
//	}
//
//
//	for (int r = 0; r < 256; ++r)
//	{
//		root = r;
//		level = 1;
//
//		for (int i = 0; i < 5; ++i)
//		{
//			for (int j = 0; j < SEARCH_TREE_SIZE; ++j)
//			{
//				searchTree[i * searchTreeRowSize + j] = 0;
//			}
//		}
//
//		searchTree[0] = 1;
//		searchTree[0 * searchTreeRowSize + 1] = root;
//
//		for (int i = 0; i < SEARCH_TREE_SIZE; i++)
//			chosenInTree[i] = 0;
//		chosenInTree[root] = true;
//		for (int i = 0; i < SEARCH_TREE_SIZE; i++)
//			visitedInCurrentSearch[i] = 0;
//
//
//
//		Enumerate(
//			root,
//			level,
//			remaring,
//			subgraphSize,
//			searchTree,
//			searchTreeRowSize,
//			chosenInTree,
//			visitedInCurrentSearch,
//			graph,
//			graphSize,
//			counter);
//	}
//
//}


////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
	srand(time(NULL));

	std::cout << SUBGRAPH_INDEX_SIZE << std::endl;
	int* counter = new int[131072];
	int* count_master = new int[131072];
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
	bool* graph_one_dim = new bool[graphSize * graphSize];
	for (int i = 0; i < graphSize; ++i)
	{
		for (int j = 0; j < graphSize; ++j)
		{
			graph_one_dim[i * graphSize + j] = graph[i][j];
		}
	}
	std::cout << std::endl << "Processing graph" << std::endl;

	ProcessGraphGPU(graph_one_dim, graphSize, count_master, 131072);
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
			/*for (int a = 0; a < 16; ++a)
			{
				if (a % 4 == 0) printf("\n");
				printf("%d", (i & (1 << (15 - a))) == 0 ? 0 : 1);
			}*/
		}
	}

	delete graph_one_dim;

	return 0;
}

