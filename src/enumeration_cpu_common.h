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
		std::atomic<int>* counter,
		int label_type);
	

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
		int offset,
		int label_type
	);
	

	int GetMaxDeg(bool* graph, int graphSize);
	
}