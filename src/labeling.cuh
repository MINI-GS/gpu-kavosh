#pragma once

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
__host__ __device__ void Label1(bool* graph, int n, bool* label);
__host__ __device__ void Label2(bool* graph, int n, bool* label);
__host__ __device__ void Label3(bool* graph, int n, bool* label);

template<class T>
__host__ __device__ void swap(int i, int j, T* tab)
{
	T temp = tab[i];
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