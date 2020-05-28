#include "enumeration_cpu_single.h"




namespace EnumerationSingle
{
	void ProcessGraph(bool* graph, int graphSize, int* counter, int counterSize, int subgraphSize = SUBGRAPH_SIZE)
	{
		const int noBlocksPerRun = 4;
		const int noThreadsPerBlock = 64;
		const int noThreadsPerRun = noBlocksPerRun * noThreadsPerBlock;
		int searchTreeRowSize = 1 + EnumerationCPU::GetMaxDeg(graph, graphSize) * (subgraphSize - 2);


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
					threads[tid] = std::thread(EnumerationCPU::EnumerateGPU,
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
		for (int i = 0; i < counterSize_d; i++)
			counter[i] = counter_d[i];

		delete[] searchTree_d;
		delete[] chosenInTree_d;
		delete[] visitedInCurrentSearch_d;
		delete[] graph_d;
		delete[] counter_d;
	}

}


