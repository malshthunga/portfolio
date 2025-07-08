## KD-Tree Nearest Neighbor Classifier (1-NN)

This Python script implements a **1-Nearest Neighbor (1-NN)** classifier using a **k-d tree** (k-dimensional tree) data structure to optimize the search process.

### Project Overview

This project was developed for an assignment at the University of Adelaide (2025), focusing on efficient classification using spatial data structures. The assignment prompt is no longer accessible, but the core objective was to:

- Build a k-d tree from a training dataset.
- Classify test data using a nearest neighbor search.
- Output the predicted labels and print the size of the first-level left/right subtrees.

### Features

- Implements recursive k-d tree construction.
- Uses a **split dimension rotation** based on student ID (even/odd rule).
- Performs **efficient nearest neighbor search** with pruning.
- Handles tie-breaking deterministically by label value.
- Compatible with whitespace-delimited `.txt` or `.dat` files for input.

- ### Example Usage

```bash
python nn_kdtree.py train.txt test.txt 0
