# Small graph  (max_id=500)

## GPU

### Config:

constexpr int SUBGRAPH_SIZE = 3; // GPU IS LIMITED TO 3
constexpr int SEARCH_TREE_SIZE = 2000;
constexpr int MAX_SUBGRAPH_SIZE = 3;
constexpr int MAX_SUBGRAPH_SIZE_SQUARED = MAX_SUBGRAPH_SIZE* MAX_SUBGRAPH_SIZE;
constexpr long SUBGRAPH_INDEX_SIZE = 1 << MAX_SUBGRAPH_SIZE_SQUARED;

constexpr int RANDOM_GRAPH_NUMBER = 1;

constexpr size_t BYTES_IN_GIGABYTE = 1'000'000'000;

### Results:

```
(using Label2)
```

Loading graph from file
Graph size: 179

Processing graph
Allocating 0.000877 GB for search tree array
Allocating 0.000046 GB for chosen in tree array
Allocating 0.000046 GB for visited in current seatch array
Allocating 0.000002 GB for counter array
Allocating 0.000032 GB for graph array
Lauching kernel
Copying couter to host
Device memory deallocation

ProcessGraphGPU duration: 73.529
graphID count
3       389850
5       648572
7       448199
39      164200
295     1057821

(��cznie 2'708'642)

Generating random graphs
.

Calculating score
graphID score
3       -1
5       -nan(ind)
7       -nan(ind)
39      -1
295     -1


```
Using Label3
```

4382    526419
4958    775162
8598    1056731
13278   228026
27030   78901
31710   43403

(��cznie 2'708'642)

```
Winiki na CPU takie same jak using Label3
```


## CPU 1 Thread

Loading graph from file
Graph size: 179

Processing graph
Lauching kernel
Copying couter to host

ProcessGraphThreading duration: 1.623
graphID count
4382    526419
4958    775162
8598    1056731
13278   228026
27030   78901
31710   43403

Generating random graphs
.

Calculating score
graphID score
4382    -1
4958    -1
8598    -1
13278   -1
27030   -1
31710   -1


## CPU multiple threads

Loading graph from file
Graph size: 179

Processing graph
Lauching kernel
Copying couter to host

ProcessGraphThreading duration: 0.46
graphID count
4382    526419
4958    775162
8598    1056731
13278   228026
27030   78901
31710   43403

Generating random graphs
.

Calculating score
graphID score
4382    -1
4958    -1
8598    -1
13278   -1
27030   -1
31710   -1



# Bigger graph  (max_id=20000)


## GPU

Loading graph from file
Graph size: 954

Processing graph
Allocating 0.003408 GB for search tree array
Allocating 0.000244 GB for chosen in tree array
Allocating 0.000244 GB for visited in current seatch array
Allocating 0.000262 GB for counter array
Allocating 0.000910 GB for graph array
Lauching kernel
kernellAssert: invalid argument


## GPU without shared memory

Loading graph from file
Graph size: 954

Processing graph
Allocating 0.003408 GB for search tree array
Allocating 0.000244 GB for chosen in tree array
Allocating 0.000244 GB for visited in current seatch array
Allocating 0.000262 GB for counter array
Allocating 0.000910 GB for graph array
Lauching kernel
Copying couter to host
Device memory deallocation

ProcessGraphGPU duration: 7532.47
graphID count
4382    59099363
4958    46066926
8598    87879603
13278   8712380
27030   3816894
31710   1125249

Generating random graphs
.

Calculating score
graphID score
4382    -1
4958    -1
8598    -1
13278   -1
27030   -1
31710   -1


## CPU 1 Thread

Loading graph from file
Graph size: 954

Processing graph
Lauching kernel
Copying couter to host

ProcessGraphThreading duration: 224.908
graphID count
4382    59099363
4958    46066926
8598    87879603
13278   8712380
27030   3816894
31710   1125249

Generating random graphs
.

Calculating score
graphID score
4382    -1
4958    -1
8598    -1
13278   -1
27030   -1
31710   -1


## CPU multiple threads

Loading graph from file
Graph size: 954

Processing graph
Lauching kernel
Copying couter to host

ProcessGraphThreading duration: 74.054
graphID count
4382    59099363
4958    46066926
8598    87879603
13278   8712380
27030   3816894
31710   1125249

Generating random graphs
.

Calculating score
graphID score
4382    -1
4958    -1
8598    -1
13278   -1
27030   -1
31710   -1