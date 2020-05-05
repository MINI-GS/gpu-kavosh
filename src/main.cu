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

#define uint unsigned int
#define ull unsigned long long

#define DEBUG

#define GRAPH_IDX(i,j) ((i)*graphSize + (j))
#define SEARCH_IDX(i,j) ((i)*searchTreeSize + (j))

__host__ __device__ void Enumerate(
		int root,
		int level,
		int remaining,
		int subgraphSize,
		int* searchTree,
		int searchTreeSize,
		bool* chosenInTree,
		bool* visitedInCurrentSearch,
		bool* graph,
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
		int* searchTree,
		int searchTreeSize,
		bool* chosenInTree,
		bool* visitedInCurrentSearch,
		bool* graph,
		int graphSize,
		int* counter
		);

__host__ __device__ void Smallest(
		int* arrH,
		int arrLen,
		bool* subgraph,
		uint* largest
		)
{
	uint val = 0;
	int iter = 0;

	for(int i = 0; i < arrLen; ++i)
	{
		int currRow = arrH[i];
		for(int j = 0; j < arrLen; ++j)
		{
			int currCol = arrH[j];
			if(subgraph[8 * currRow + currCol])
			{
				val += (1<<iter);
			}
			++iter;
		}
	}

	if(val > *largest) *largest = val;
}

__host__ __device__ void Horisontal(
		int* arrH,
		int* arrV,
		int arrLen,
		int level,
		bool* subgraph,
		uint* largest
		)
{
	if(level == arrLen)
	{
		Smallest(arrH, arrLen, subgraph, largest);
		return;
	}
	for(int i = level; i < arrLen; ++i)
	{
		int tmp = arrH[level];
		arrH[level] = arrH[i];
		arrH[i] = tmp;
		Horisontal(arrH, arrV, arrLen, level + 1, subgraph, largest);
		tmp = arrH[level];
		arrH[level] = arrH[i];
		arrH[i] = tmp;
	}
}


__host__ __device__ void Revolve(
		int n,
		int left,
		int right,

		int root,
		int level,
		int remaining,
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
	int* tab = searchTree + (level*searchTreeSize);

	if(n == 0)
	{
		for(int i = left; i < right; ++i)
		{
			chosenInTree[tab[i]] = false;
		}

		Enumerate(
				root,
				level + 1,
				remaining,
				subgraphSize,
				searchTree,
				searchTreeSize,
				chosenInTree,
				visitedInCurrentSearch,
				graph,
				graphSize,
				counter);

	}
	else if(n == right - left)
	{
		for(int i = left; i < right; ++i)
		{
			chosenInTree[tab[i]] = true;
		}

		Enumerate(
				root,
				level + 1,
				remaining,
				subgraphSize,
				searchTree,
				searchTreeSize,
				chosenInTree,
				visitedInCurrentSearch,
				graph,
				graphSize,
				counter);

		for(int i = left; i < right; ++i)
		{
			chosenInTree[tab[i]] = false;
		}
	}
	else
	{
		chosenInTree[tab[left]] = false;
		Revolve(n, left+1, right,
				root,
				level,
				remaining,
				subgraphSize,
				searchTree,
				searchTreeSize,
				chosenInTree,
				visitedInCurrentSearch,
				graph,
				graphSize,
				counter);

		chosenInTree[tab[left]] = true;
		RevolveR(n - 1, left+1, right,
				root,
				level,
				remaining,
				subgraphSize,
				searchTree,
				searchTreeSize,
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
		int* searchTree,
		int searchTreeSize,
		bool* chosenInTree,
		bool* visitedInCurrentSearch,
		bool* graph,
		int graphSize,
		int* counter
		)
{
	int* tab = searchTree + (level*searchTreeSize);

	if(n == 0)
	{
		for(int i = left; i < right; ++i)
		{
			chosenInTree[tab[i]] = false;
		}

		Enumerate(
				root,
				level + 1,
				remaining,
				subgraphSize,
				searchTree,
				searchTreeSize,
				chosenInTree,
				visitedInCurrentSearch,
				graph,
				graphSize,
				counter);
	}
	else if(n == right - left)
	{
		for(int i = left; i < right; ++i)
		{
			chosenInTree[tab[i]] = true;
		}

		Enumerate(
				root,
				level + 1,
				remaining,
				subgraphSize,
				searchTree,
				searchTreeSize,
				chosenInTree,
				visitedInCurrentSearch,
				graph,
				graphSize,
				counter);

		for(int i = left; i < right; ++i)
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
				searchTreeSize,
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
				searchTreeSize,
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
		int* searchTree,
		int searchTreeSize,
		bool* chosenInTree,
		bool* visitedInCurrentSearch,
		bool* graph,
		int graphSize)
{
	searchTree[SEARCH_IDX(level,0)] = 0;
	for(int i = 1; i <= searchTree[SEARCH_IDX(level-1,0)]; ++i)
	{
		if(chosenInTree[searchTree[SEARCH_IDX(level-1,i)]])
		{
			int parent = searchTree[SEARCH_IDX(level-1,i)];

			for(int a = root + 1; a < graphSize; ++a)
			{
				if(!visitedInCurrentSearch[a] && a != parent)
				{
					if(graph[GRAPH_IDX(parent,a)] || graph[GRAPH_IDX(a,parent)])
					{
						int ind = ++searchTree[SEARCH_IDX(level,0)];
						searchTree[SEARCH_IDX(level,ind)] = a;
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
		int* searchTree,
		int searchTreeSize,
		bool* chosenInTree,
		bool* visitedInCurrentSearch,
		bool* graph,
		int graphSize,
		int* counter)
{
	if(remaining == 0)
	{
		int arrH[8];
		int arrV[8];

		for(int i = 0; i < 8; ++i)
		{
			arrH[i] = arrV[i] = i;
		}
		bool subgraph[64];
		for(int i = 0; i < 64; ++i) subgraph[i] = false;

		int chosenVerts[8];
		int iter = 0;
		for(int i = 0; i < graphSize; ++i)
		{
			if(chosenInTree[i]) chosenVerts[iter++] = i;
		}

		for(int i = 0; i < subgraphSize; ++i)
		{
			int vert = chosenVerts[i];
			for(int j = 0; j < subgraphSize; ++j)
			{
				int vert2 = chosenVerts[j];
				if (vert2 == vert) continue;
				if(graph[GRAPH_IDX(vert,vert2)])
				{
					subgraph[8 * i + j] = true;
				}

			}
		}


		return;
	}


	InitChildSet(
			root,
			level,
			searchTree,
			searchTreeSize,
			chosenInTree,
			visitedInCurrentSearch,
			graph,
			graphSize);

	for(int k = 1; k <= remaining; ++k)
	{
		if(searchTree[SEARCH_IDX(level,0)] < k)
		{
			return;
		}

		Revolve(
			k,
			1,
			searchTree[SEARCH_IDX(level,0)] + 1,
			root,
			level,
			remaining - k,
			subgraphSize,
			searchTree,
			searchTreeSize,
			chosenInTree,
			visitedInCurrentSearch,
			graph,
			graphSize,
			counter);
	}

	for(int i = 1; i <= searchTree[SEARCH_IDX(level,0)]; ++i)
	{
		visitedInCurrentSearch[searchTree[SEARCH_IDX(level,i)]] = false;
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
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < graphSize)
	{
		int * searchTreeRoot = searchTree + (tid * searchTreeSize * subgraphSize);
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
		__syncthreads();
	}
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char ** argv)
{
	int searchTreeSize = 10;
	int subgraphSize = 4;

	int* searchTree_d;
	checkCudaErrors(cudaMalloc((void**)&searchTree_d,  20000 * sizeof(int)));
	checkCudaErrors(cudaMemset(searchTree_d, 0, 20000 * sizeof(int)));

	bool* chosenInTree_d;
	checkCudaErrors(cudaMalloc((void**)&chosenInTree_d,  2000 * sizeof(bool)));
	checkCudaErrors(cudaMemset(chosenInTree_d, 0, 2000 * sizeof(bool)));

	bool* visitedInCurrentSearch_d;
	checkCudaErrors(cudaMalloc((void**)&visitedInCurrentSearch_d,  2000 * sizeof(bool)));
	checkCudaErrors(cudaMemset(visitedInCurrentSearch_d, 0, 2000 * sizeof(bool)));

	bool* graph = new bool[49];

	int graphSize = 7;

	graph[GRAPH_IDX(0,1)] = true;
	graph[GRAPH_IDX(0,2)] = true;
	graph[GRAPH_IDX(1,5)] = true;
	graph[GRAPH_IDX(2,6)] = true;
	graph[GRAPH_IDX(2,4)] = true;
	graph[GRAPH_IDX(2,5)] = true;
	graph[GRAPH_IDX(3,2)] = true;
	graph[GRAPH_IDX(4,3)] = true;
	graph[GRAPH_IDX(4,5)] = true;
	graph[GRAPH_IDX(4,2)] = true;
	graph[GRAPH_IDX(4,0)] = true;
	graph[GRAPH_IDX(5,6)] = true;
	graph[GRAPH_IDX(5,3)] = true;
	graph[GRAPH_IDX(6,1)] = true;

	bool* graph_d;
	checkCudaErrors(cudaMalloc((void**)&graph_d,  graphSize * sizeof(bool)));
	checkCudaErrors(cudaMemcpy(graph_d, graph , graphSize * sizeof(bool), cudaMemcpyHostToDevice));

	int* counter = new int[131071];

	int* counter_d;
	checkCudaErrors(cudaMalloc((void**)&counter_d,  131071 * sizeof(int)));
	checkCudaErrors(cudaMemset(counter_d, 0, 131071 * sizeof(int)));

	uint* largest_d;
	checkCudaErrors(cudaMalloc((void**)&largest_d,  128 * sizeof(uint)));

	EnumerateGPU<<<1,1>>>(
			subgraphSize,
			searchTree_d,
			searchTreeSize,
			chosenInTree_d,
			visitedInCurrentSearch_d,
			graph_d,
			graphSize,
			counter_d);

	cudaDeviceSynchronize();
	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess)
	{
		fprintf(stderr, "kernelAssert: %s\n", cudaGetErrorString(code));
		if (abort) exit(code);
	}

	checkCudaErrors(cudaMemcpy((void*)(counter), (void*)counter_d, 131071*sizeof(int), cudaMemcpyDeviceToHost));

	for(uint i = 0; i<131071; ++i)
	{
		if(counter[i] != 0)
		{
			printf("\n%d %d", i, counter[i]);
			for(int a = 0; a < 16; ++a)
			{
				if(a%4 == 0) printf("\n");
				printf("%d", (i & (1 << (15 - a))) == 0 ? 0 : 1);
			}
		}
	}



	printf("\nHELLO");
}

