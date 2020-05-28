#pragma once

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

#include <vector>
#include <atomic>
#include <thread>

#include "config.h"
#include "labeling.cuh"


#include <map>

#define uint unsigned int
#define ull unsigned long long



namespace EnumerationCPU
{
	template<class T>
	void swap(int i, int j, T* tab)
	{
		T temp = tab[i];
		tab[i] = tab[j];
		tab[j] = temp;
	}

	template<class T>
	void reverse(int i, int j, T* tab)
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
	bool NextPermutation(int first, int last, T* tab)
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

	void InitChildSet(
		int root,
		int level,
		int* searchTree,
		int searchTreeRowSize,
		bool* chosenInTree,
		bool* visitedInCurrentSearch,
		bool* graph,
		int graphSize);
	

	void Label2(bool* graph, int n, bool* label);
	


	void Enumerate(
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
		std::atomic<int>* counter);
	

	void EnumerateGPU(
		int subgraphSize,
		int* searchTree,
		int searchTreeSize,
		bool* chosenInTree,
		bool* visitedInCurrentSearch,
		bool* graph,
		int graphSize,
		std::atomic<int>* counter,
		int tid,
		int offset
	);
	

	int GetMaxDeg(bool* graph, int graphSize);
	
}