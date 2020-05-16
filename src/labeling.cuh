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




__host__ __device__ bool get_edge(int vertex1, int vertex2, bool* graph);
__host__ __device__ void set_edge(int vertex1, int vertex2, bool* graph, bool value);
__host__ __device__ void Label(bool* graph, int n, bool* label);
__host__ __device__ void Label3(bool* graph, int n, bool* label);

