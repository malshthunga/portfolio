STUDENT_ID = 'a1895261'
DEGREE = 'UG'

import numpy as np
import pandas as pd
import sys

class KDNode:
    def __init__(self, point=None, label=None, split_dim=None, split_val=None, left=None, right=None):
        self.point = point
        self.label = label
        self.split_dim = split_dim
        self.split_val = split_val
        self.left = left
        self.right = right

def build_kdtree(points, depth=0, start_dim=0, count_split=False):
    if not points:
        return None
    k = len(points[0][0])

    if len(points) == 1:
        point, label = points[0]
        axis = (depth + start_dim) % k
        split_val = point[axis]
        return KDNode(point, label, split_dim=axis, split_val=split_val)

    axis = (depth + start_dim) % k
    is_odd = int(STUDENT_ID[-1]) % 2 == 1

    axis_values = [p[0][axis] for p in points]
    median_val = float(np.median(axis_values))

    # Find first point with feature[axis] == median_val in original order
    median_idx = None
    for idx, (pt, lb) in enumerate(points):
        if float(pt[axis]) == median_val:
            median_idx = idx
            break
    if median_idx is None:
        median_idx = min(range(len(points)), key=lambda i: abs(float(points[i][0][axis]) - median_val))
    median_point, median_label = points[median_idx]

    left_points = []
    right_points = []
    for idx, (pt, lb) in enumerate(points):
        if idx == median_idx:
            continue
        if float(pt[axis]) < median_val:
            left_points.append((pt, lb))
        elif float(pt[axis]) > median_val:
            right_points.append((pt, lb))
        else:
            if is_odd:
                left_points.append((pt, lb))
            else:
                right_points.append((pt, lb))

    left_child = build_kdtree(left_points, depth + 1, start_dim)
    right_child = build_kdtree(right_points, depth + 1, start_dim)
    node = KDNode(median_point, median_label, split_dim=axis, split_val=median_val, left=left_child, right=right_child)

    if count_split and depth == 0:
        return node, len(left_points), len(right_points)
    return node

def euclidean_distance(a, b):
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))

def nearest_neighbor(node, query, start_dim, depth=0, best=None, best_dist=float('inf')):
    if node is None:
        return best, best_dist

    axis = (depth + start_dim) % len(query)
    dist = euclidean_distance(query, node.point)

    if (best is None) or (dist < best_dist) or (dist == best_dist and node.label < best.label):
        best = node
        best_dist = dist

    # Decide which branch to search first - respect "less than or equal" rule for odd student ID
    if query[node.split_dim] <= node.split_val:
        first_branch, second_branch = node.left, node.right
    else:
        first_branch, second_branch = node.right, node.left

    best, best_dist = nearest_neighbor(first_branch, query, start_dim, depth + 1, best, best_dist)

    # Check if we need to explore the other branch
    if (query[node.split_dim] - node.split_val) ** 2 < best_dist:
        best, best_dist = nearest_neighbor(second_branch, query, start_dim, depth + 1, best, best_dist)

    return best, best_dist

def main():
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    start_dim = int(sys.argv[3])

    train_df = pd.read_csv(train_file, sep=r'\s+')
    test_df = pd.read_csv(test_file, sep=r'\s+')

    X_train = train_df.iloc[:, :-1].values.tolist()
    y_train = train_df.iloc[:, -1].values.tolist()
    train_points = list(zip(X_train, y_train))
    X_test = test_df.values.tolist()

    # Build tree, print split info
    result = build_kdtree(train_points, 0, start_dim, count_split=True)
    if isinstance(result, tuple):
        root, left_count, right_count = result
        print('.' * start_dim + f"l{left_count}")
        print('.' * start_dim + f"r{right_count}")
    else:
        root = result

    for test_vec in X_test:
        nn, _ = nearest_neighbor(root, test_vec, start_dim)
        print(int(nn.label))  # Output as integer as per sample output

if __name__ == "__main__":
    main()
