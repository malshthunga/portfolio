
This Python script implements three classical search algorithms to solve grid-based pathfinding problems:

- **Breadth-First Search (BFS)**
- **Uniform Cost Search (UCS)**
- **A Search (with Manhattan or Euclidean heuristics)**

This was developed as part of a university assignment (University of Adelaide, 2025). The assignment description is no longer accessible, but the objective was to implement and compare search algorithms for finding the optimal path on a grid map, taking into account obstacles and elevation.

The script takes in a `.txt` file representing the map, a selected algorithm, and an optional heuristic type, and outputs the final path on the grid.

How to run 
python pathfinder.py [mode] [mapfile.txt] [algorithm] [heuristic]
