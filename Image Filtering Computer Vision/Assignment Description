#  Computer Vision Assignment 1 – Image Filtering and Processing

This repository contains my implementation and write-up for **Assignment 1** of the **Computer Vision 2025** course at the **University of Adelaide**. The goal of this assignment was to research, implement, and test fundamental image filtering operations using Python and core libraries such as NumPy, OpenCV, and Matplotlib.

---

##  Objectives

- Understand how images are stored and processed in memory.
- Gain hands-on experience with **image filters** (brightness, contrast, gamma, thresholding).
- Learn to implement **convolution** from scratch and apply filters (Gaussian, Sobel, LoG).
- Work with image **sampling**, **resizing**, and **pyramid creation**.
- Explore the importance of **scale**, **rotation**, and **blur** in computer vision tasks.

---

##  Structure

- `Assignment_1_Notebook.ipynb`: Main Jupyter notebook containing all code, analysis, and visual results.
- `a1code.py`: Core module with custom image filtering functions.
- `images/`: Folder containing test images used in experiments.
- `outputs/`: Folder to store transformed/filtered images.

---

##  Key Implementations

###  Point Processing Functions
- `adjust_brightness(image, value)`
- `contrast_stretching(image)`
- `gamma_correction(image, gamma)`
- `thresholding(image, threshold)`
- `logarithmic_transformation(image)`
- `power_law_transformation(image, gamma)`

###  Spatial Processing & Filtering
- `crop()`, `resize()`, `change_contrast()`, `greyscale()`, `binary()`
- `conv2D()`: Custom 2D convolution function (handles border padding + flipped kernel)
- `conv()`: RGB-compatible convolution
- `gauss2D()`, `LoG2D()`: Gaussian and Laplacian of Gaussian filter generation
- `sobel_x`, `sobel_y`: Horizontal and vertical edge detection

###  Experiments
- Compared filtered vs unfiltered images
- Visualized:
  - Gradient magnitude
  - Image pyramids
  - Edge maps with Sobel filters
  - Blob detection with LoG filters
- Analyzed effects of:
  - Gaussian blur before downsampling
  - Filter size and sigma on blur/sharpness
  - Image rotation and LoG invariance

---

##  Sample Images Used
- `mandrill.jpg` – high color intensity, good for RGB transformations.
- `cat.jpg` – rich in textures and edges, ideal for contrast and Sobel filters.
- `elephant.jpg`, `parrot.jpg`, `hummingbird.jpg`, `whipbird.jpg` – varied content and detail levels for robust testing.

---

##  Insights & Observations

- Downsampling without prior blurring introduces aliasing artifacts.
- Larger Gaussian kernels smooth more detail, acting as low-pass filters.
- Sobel filters are effective for edge detection and directional gradients.
- LoG filters are rotationally symmetric and useful for blob detection.

---

##  Skills Demonstrated

- Image filtering & enhancement
- 2D convolution from scratch
- Python data manipulation & visualization
- Scientific analysis and interpretation
- Use of OpenCV, NumPy, and Matplotlib in computer vision

---

##  Technologies Used

- Python 3.x
- NumPy
- OpenCV
- Matplotlib
- scikit-image
- Jupyter Notebook

---

##  Educational Context

This was an assessed assignment for the course *Computer Vision 2025* at the **University of Adelaide**, completed in **Semester 1**. The assignment reinforces key concepts needed for advanced topics like **Convolutional Neural Networks (CNNs)** and **feature extraction** in computer vision.

---


