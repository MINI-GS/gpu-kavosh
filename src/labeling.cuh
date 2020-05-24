#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <assert.h>
#include <random>
#include <float.h>



bool get_edge(int vertex1, int vertex2, bool* graph);
void set_edge(int vertex1, int vertex2, bool* graph, bool value);
void Label(bool* graph, int n, bool* label);
void Label3(bool* graph, int n, bool* label);

