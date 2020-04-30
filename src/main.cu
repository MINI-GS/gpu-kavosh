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


__device__ void RevolveR(int n, bool* tab, int tablen, int left, int right);

__device__ void Revolve(int n, bool* tab, int tablen, int left, int right)
{
	if(n == 0)
	{
		for(int i = left; i < right; ++i)
		{
			tab[i] = false;
		}

		//Enumerate lvl deeper

	}
	else if(n == right - left)
	{
		for(int i = left; i < right; ++i)
		{
			tab[i] = true;
		}

		//Enumerate lvl deeper
	}
	else
	{
		tab[left] = false;
		Revolve(n, tab, tablen, left+1, right);

		tab[left] = true;
		RevolveR(n - 1, tab, tablen, left+1, right);
	}
}

__device__ void RevolveR(int n, bool* tab, int tablen, int left, int right)
{
	if(n == 0)
	{
		for(int i = left; i < right; ++i)
		{
			tab[i] = false;
		}

		//Enumerate lvl deeper
	}
	else if(n == right - left)
	{
		for(int i = left; i < right; ++i)
		{
			tab[i] = true;
		}

		//Enumerate lvl deeper
	}
	else
	{
		tab[right - 1] = false;
		RevolveR(n, tab, tablen, left, right - 1);

		tab[right - 1] = true;
		Revolve(n - 1, tab, tablen, left, right - 1);
	}
}

__device__ void Enumerate(
		int root,
		int level,
		int remaring,
		int subgraphSize,
		int** searchTree,
		bool* visitedInCurrentSearch,
		int** graph)
{
	if(remaining == 0)
	{
		// isomorphism
		return;
	}

	// initChildSet

	for(int k = 1; k <= remaring; ++k)
	{
		if(searchTree[level][0] < k)
		{
			return;
		}

		//build next level

		//Revolve
	}



}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char ** argv)
{
}

