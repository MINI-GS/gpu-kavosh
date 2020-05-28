#pragma once
#include "enumeration_cpu_common.h"

namespace EnumerationSingle
{
	void ProcessGraph(bool* graph, int graphSize, int* counter, int counterSize, int subgraphSize, int label_type);
}