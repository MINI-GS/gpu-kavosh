#include "labeling.cuh"

#include "config.h"

// to change, dynamic allocation?


__host__ __device__ bool get_edge(int vertex1, int vertex2, bool* graph)
{
	return graph[MAX_SUBGRAPH_SIZE * vertex1 + vertex2];
}

__host__ __device__  void set_edge(int vertex1, int vertex2, bool* graph, bool value)
{
	graph[MAX_SUBGRAPH_SIZE * vertex1 + vertex2] = value;
}


 __host__ __device__ void permute1(int* vertex_label, bool* graph, bool* label, int l, int r)
{
	// end of heap
	if (l == r)
	{
		int result = 0; // 0 - continue, 1 - current is beter, 2 - current is worse
		for (int i = 0; (i < SUBGRAPH_SIZE) && !result; i++)
			for (int j = 0; (j < SUBGRAPH_SIZE) && !result; j++)
			{
				// if current permutation is better
				if (get_edge(vertex_label[i], vertex_label[j], graph) && !get_edge(i, j, label))
					result = 1;
				else if (!get_edge(vertex_label[i], vertex_label[j], graph) && get_edge(i, j, label))
					result = 2;
			}

		// if current is not better
		if (result != 1)
			return;

		// save current
		for (int i = 0; (i < SUBGRAPH_SIZE); i++)
			for (int j = 0; (j < SUBGRAPH_SIZE); j++)
			{
				bool edge_value = get_edge(vertex_label[i], vertex_label[j], graph);
				set_edge(i, j, label, edge_value);
			}

		return;
	}

	// Permutations  
	for (int i = l; i <= r; i++)
	{

		// Swapping  
		int tmp = vertex_label[l];
		vertex_label[l] = vertex_label[i];
		vertex_label[i] = tmp;

		// Recursion called  
		permute1(vertex_label, graph, label, l + 1, r);

		// Backtrack  
		tmp = vertex_label[l];
		vertex_label[l] = vertex_label[i];
		vertex_label[i] = tmp;
	}
}

__host__ __device__ void Label1(bool* graph, int n, bool* label)
{
	//assert SUBGRAPH_SIZE == n

	int vertex_label[SUBGRAPH_SIZE];
	for (int i = 0; i < SUBGRAPH_SIZE; i++)
		vertex_label[i] = i;
	permute1(vertex_label, graph, label, 0, n - 1);

	return void();
}

/// following code is based on std::nextpermutation
/// posible implememtation from cppreference
/// true if the function could rearrange the
/// object as a lexicographicaly greater permutation.
/// Otherwise, the function returns false to indicate
/// that the arrangement is not greater than the previous,
/// but the lowest possible(sorted in ascending order).
__host__ __device__ void Label2(bool* graph, int n, bool* label)
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


__host__ __device__ void permute3(int* vertex_label, bool* graph, bool* label, int l, int r)
{
	int indexes[SUBGRAPH_SIZE];
	for (int i = 0; i < SUBGRAPH_SIZE; i++)
		indexes[i] = i;
	while (true)
	{
		if (l < 0)
			return;

		// end of heap
		if (l == r)
		{
			int result = 0; // 0 - continue, 1 - current is beter, 2 - current is worse
			for (int i = 0; (i < SUBGRAPH_SIZE) && !result; i++)
				for (int j = 0; (j < SUBGRAPH_SIZE) && !result; j++)
				{
					// if current permutation is better
					if (get_edge(vertex_label[i], vertex_label[j], graph) && !get_edge(i, j, label))
						result = 1;
					else if (!get_edge(vertex_label[i], vertex_label[j], graph) && get_edge(i, j, label))
						result = 2;
				}

			// if current is not better
			if (result != 1)
			{
				l--;
				continue;
			}
				

			// save current
			for (int i = 0; (i < SUBGRAPH_SIZE); i++)
				for (int j = 0; (j < SUBGRAPH_SIZE); j++)
				{
					bool edge_value = get_edge(vertex_label[i], vertex_label[j], graph);
					set_edge(i, j, label, edge_value);
				}

			l--;
			continue;
		}

		if (indexes[l] == l)
		{
			// Swapping  
			int tmp = vertex_label[l];
			vertex_label[l] = vertex_label[indexes[l]];
			vertex_label[indexes[l]] = tmp;
			
			indexes[l]++;

			l++;
			continue;
		}
		else if(indexes[l] <= r)
		{
			int tmp = vertex_label[l];
			vertex_label[l] = vertex_label[indexes[l]-1];
			vertex_label[indexes[l]-1] = tmp;

			tmp = vertex_label[l];
			vertex_label[l] = vertex_label[indexes[l]];
			vertex_label[indexes[l]] = tmp;

			indexes[l]++;

			l++;
			continue;
		}
		else if (indexes[l] == r+1)
		{
			int tmp = vertex_label[l];
			vertex_label[l] = vertex_label[indexes[l] - 1];
			vertex_label[indexes[l] - 1] = tmp;

			indexes[l]=l;
			l--;
			continue;
		}
		
		printf("To nie powinno sie zdarzyc\n");
	}


}

__host__ __device__ void Label3(bool* graph, int n, bool* label)
{
	//assert SUBGRAPH_SIZE == n
	if (n != SUBGRAPH_SIZE)
	{
		printf("Bad n in labeling\n");
		return;
	}

	int vertex_label[SUBGRAPH_SIZE];
	for (int i = 0; i < SUBGRAPH_SIZE; i++)
		vertex_label[i] = i;
	permute3(vertex_label, graph, label, 0, SUBGRAPH_SIZE - 1);

	return void();
}