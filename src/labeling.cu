#include "labeling.cuh"

// to change, dynamic allocation?
constexpr int SUBGRAPH_SIZE = 4;


__host__ __device__ bool get_edge(int vertex1, int vertex2, bool* graph)
{
	//printf("%d:%d\n", vertex1, vertex2);
	return graph[8 * vertex1 + vertex2];
}


__host__ __device__ void set_edge(int vertex1, int vertex2, bool* graph, bool value)
{
	graph[8 * vertex1 + vertex2] = value;
}


__host__ __device__ void permute(int* vertex_label, bool* graph, bool* label, int l, int r)
{
	// end of heap
	if (l == r)
	{
		int result = 0; // 0 - continue, 1 - current is beter, 2 - current is worse
		for (int i = 0; (i < SUBGRAPH_SIZE) && !result; i++)
			for (int j = 0; (j < SUBGRAPH_SIZE) && !result; j++)
			{
				//printf("vl:%d:%d:%d:%d\n", vertex_label[0], vertex_label[1], vertex_label[2], vertex_label[3]);
				//printf("%d:%d:%d:%d\n", i, j, vertex_label[i], vertex_label[j]);
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
				//printf("vl:%d:%d:%d:%d\n", vertex_label[0], vertex_label[1], vertex_label[2], vertex_label[3]);
				//printf("aa:%d:%d:%d:%d\n", i, j, vertex_label[i], vertex_label[j]);
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
		permute(vertex_label, graph, label, l + 1, r);

		// Backtrack  
		tmp = vertex_label[l];
		vertex_label[l] = vertex_label[i];
		vertex_label[i] = tmp;
	}
}

__host__ __device__ void Label(bool* graph, int n, bool* label)
{
	//assert SUBGRAPH_SIZE == n

	int vertex_label[SUBGRAPH_SIZE];
	for (int i = 0; i < SUBGRAPH_SIZE; i++)
		vertex_label[i] = i;
	//printf("vl0:%d:%d:%d:%d\n", vertex_label[0], vertex_label[1], vertex_label[2], vertex_label[3]);

	permute(vertex_label, graph, label, 0, n - 1);

	return void();
}
