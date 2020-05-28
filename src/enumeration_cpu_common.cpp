#include "enumeration_cpu_common.h"

void EnumerationCPU::InitChildSet(int root, int level, int* searchTree, int searchTreeRowSize, bool* chosenInTree, bool* visitedInCurrentSearch, bool* graph, int graphSize)
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

void EnumerationCPU::Label2(bool* graph, int n, bool* label)
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

void EnumerationCPU::Enumerate(int root, int level, int remaining, int subgraphSize, int* searchTree, int searchTreeRowSize, bool* chosenInTree, bool* visitedInCurrentSearch, bool* graph, int graphSize, std::atomic<int>* counter, int label_type)
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

		switch (label_type)
		{
		case 1:
			Label1(subgraph, subgraphSize, label);
			break;
		case 2:
			Label2(subgraph, subgraphSize, label);
			break;
		case 3:
			Label3(subgraph, subgraphSize, label);
			break;
		default:
			break;
		}
		
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
				counter,
				label_type);

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

void EnumerationCPU::EnumerateGPU(int subgraphSize, int* searchTree, int searchTreeSize, bool* chosenInTree, bool* visitedInCurrentSearch, bool* graph, int graphSize, std::atomic<int>* counter, int tid, int offset, int label_type)
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
			counter,
			label_type);
	}

}

int EnumerationCPU::GetMaxDeg(bool* graph, int graphSize)
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
