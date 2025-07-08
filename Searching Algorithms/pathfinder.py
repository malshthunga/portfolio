STUDENT_ID = 'a1895261'
DEGREE='UG'

import sys #used for ucs 
import math
import numpy as np
import heapq
from typing import List, Tuple, Optional
from collections import deque
from itertools import count

def load_map_data(file_path):
    with open(file_path, "r") as f:
        lines = f.read().strip().splitlines()
        rows, cols = map(int, lines[0].split())
        start = tuple(int(x) - 1 for x in lines[1].split())
        goal = tuple(int(x) - 1 for x in lines[2].split())
        grid = []
        for line in lines[3:]:
            grid.append([int(x) if x != 'X' else -1 for x in line.split()])
        return (rows, cols), start, goal, grid
    

def bfs(grid, start, end):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Directions for movement: up, down, left, right
    queue = deque([(start, [start])])  # Queue contains tuples of (current node, path to current node)
    visited = {start}  # Set of visited nodes to avoid re-visiting
    
    while queue:
        current, path = queue.popleft()  # Dequeue the front element
        if current == end:  # If the current node is the end, return the path
            return path
        for d in directions:  # Explore neighbors
            new_x, new_y = current[0] + d[0], current[1] + d[1]
            # Check if the new position is within bounds, not an obstacle, and not visited
            if 0 <= new_x < len(grid) and 0 <= new_y < len(grid[0]) and grid[new_x][new_y] != -1 and (new_x, new_y) not in visited:
                visited.add((new_x, new_y))  # Mark the new position as visited
                queue.append(((new_x, new_y), path + [(new_x, new_y)]))  # Enqueue the new node with the updated path
    
    return None  # If no path found



def ucs(rows, cols, start_row, start_col, end_row, end_col, grid):
    visited = set()
    queue = []
    parent = {}
    cost = {}
    counter = count()
    
    start = (start_row, start_col)
    goal = (end_row, end_col)
    heapq.heappush(queue, (0, next(counter), start))
    cost[start] = 0

    while queue:
        current_cost, _, current = heapq.heappop(queue)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return reconstruct_path(parent, goal)

        for neighbor in get_neighbors(current, grid):
            new_cost = current_cost + calculate_cost(current, neighbor, grid)
            if neighbor not in cost or new_cost < cost[neighbor]:
                cost[neighbor] = new_cost
                parent[neighbor] = current
                heapq.heappush(queue, (new_cost, next(counter), neighbor))

    return []

def reconstruct_path(parent, end):
    path=[]
    while end in parent:
        path.append(end)
        end = parent[end]
    path.append(end)
    return path[::-1]

def get_neighbors(position,grid):
    for delta_x, delta_y in [(-1,0),(1,0),(0,-1),(0,1)]:
        new_x, new_y  = position[0] + delta_x, position[1] + delta_y
        if 0 <= new_x < len(grid) and 0 <= new_y < len(grid[0]) and grid[new_x][new_y] != -1:
            yield(new_x, new_y)

def calculate_cost(current_pos, next_pos, grid):
    elevation_current = grid[current_pos[0]][current_pos[1]]
    elevation_next = grid[next_pos[0]][next_pos[1]]
    elevation_difference = elevation_next - elevation_current
    return 1 + max(0, elevation_difference)

def heuristic(current_pos,goal_pos,method):
    if method == 'manhattan':
        return abs(current_pos[0] -goal_pos[0] )+ abs(current_pos[1]- goal_pos[1])
    elif method == 'euclidean':
        return math.hypot(current_pos[0] -goal_pos[0], current_pos[1] - goal_pos[1])
    return 0


def astar(grid, start, end, heuristic_type):
    open_set = []

    # Initial cost is 0; est_total is cost + heuristic to end
    h = heuristic(start, end, heuristic_type)
    heapq.heappush(open_set, (h, 0, start))

    came_from = {}
    cost = {start: 0}
    counter = count()

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current == end:
            return reconstruct_path(came_from, current)

        for delta_x, delta_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + delta_x, current[1] + delta_y)
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][neighbor[1]] != -1:
                move_cost = calculate_cost(current, neighbor, grid)
                new_cost = cost[current] + move_cost

                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    came_from[neighbor] = current
                    est_total = new_cost + heuristic(neighbor, end, heuristic_type)
                    heapq.heappush(open_set, (est_total, next(counter), neighbor))

    return None



def print_path(grid, path):
    if path is None:
        print("null")
        return
    for i,j in path:
        grid[i][j] = '*'
    for row in grid:
        print(' '.join(str(x).replace('-1','X') for x in row))

#main() 

def main():
    if len(sys.argv) < 4:
        print("Usage: python pathfinder.py [mode] [mapfile] [algorithm] [heuristic]")
        return
    mode, file_path, algo = sys.argv[1:4]
    heuristic_type = sys.argv[4] if len(sys.argv) > 4 else None
    (rows, cols), start, goal, grid = load_map_data(file_path)
    path = None
    if algo == 'bfs':
        path = bfs(grid, start, goal)
    elif algo == 'ucs':
        path = ucs(rows, cols, *start, *goal, grid)
    elif algo == 'astar' and heuristic_type:
        path = astar(grid, start, goal, heuristic_type)
    print_path(grid, path)

if __name__ == "__main__":
    main()




