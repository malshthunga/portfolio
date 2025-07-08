#  Feature Matching in Computer Vision

This repository contains my implementation and results for a **Feature Matching** assignment completed for the **Computer Vision 2025** course at the **University of Adelaide**. The project explores keypoint detection, descriptor computation, and feature matching using classical computer vision techniques.

---

##  Objectives

- Understand how to **detect interest points** in an image.
- Implement and compare **keypoint detectors**: Harris and Difference of Gaussian (DoG).
- Compute **local descriptors** using patch-based extraction.
- Match keypoints across images using **L2 norm** and **mutual nearest neighbor filtering**.
- Visualize and evaluate feature correspondences between image pairs.

---

##  Implemented Components

###  Keypoint Detection
- **Harris Corner Detector**: Based on image gradients and structure tensor.
- **Difference of Gaussians (DoG)**: Used to detect blob-like structures.

###  Descriptor Computation
- Local **square patches** extracted around keypoints.
- Patch normalization to reduce sensitivity to brightness and contrast.

###  Feature Matching
- **L2 Distance** (Euclidean) to compare descriptors.
- **Mutual Nearest Neighbor** check to improve match reliability.
- Matched pairs are visualized using line overlays.

---

##  File Structure

- `feature_matching.ipynb`: Main notebook with code, results, and discussion.
- `src/` folder (optional): Contains modular code for Harris, DoG, matching, and visualization.
- `images/`: Folder containing test images (e.g. `notre_dame`, `mt_rushmore`, `eiffel`).
- `outputs/`: Folder storing visualized matches.

---

##  Sample Experiments

Tested on:
- `notre_dame` pair
- `mt_rushmore` pair
- `eiffel` pair

Each pair demonstrates:
- Detected keypoints overlayed on the image
- Top matches before and after mutual filtering
- Performance differences between Harris and DoG

---

##  Key Insights

- DoG detects more blob-like features; Harris detects corners.
- Patch size and normalization are critical to match quality.
- Mutual nearest neighbor filtering significantly reduces false matches.

---

##  Skills Demonstrated

- Classical keypoint detection (Harris, DoG)
- Descriptor extraction and comparison
- Matching strategies (NN, mutual NN)
- Python and NumPy for image processing
- Matplotlib for visualization

---

##  Technologies Used

- Python 3.x
- NumPy
- OpenCV
- Matplotlib
- Jupyter Notebook

---

##  Educational Context

This project was completed as part of the **Computer Vision 2025** course at the **University of Adelaide**. It reflects foundational skills in classical computer vision, relevant for tasks
