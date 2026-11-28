# 1. Introduction

Images are often displayed at a different aspect ratio from the one they were taken in. The two standard ways to change the size of an image both ignore what the image shows. **Scaling** resamples every pixel by the same factor, so when the aspect ratio changes, people get thinner, circles become ellipses and buildings lean. **Cropping** keeps the content undistorted but throws away whatever lies outside the window, which is a poor trade when the important content is spread across the frame.

Avidan and Shamir [1] proposed a third option, *seam carving*. A vertical seam is a connected path of pixels from the top row to the bottom row with exactly one pixel per row, where the column changes by at most one between consecutive rows. Removing a vertical seam makes the image one pixel narrower; because the removed pixel can sit in a different column in every row, the seam can bend around important objects and pass through uniform areas such as sky, water or grass. Each pixel is given an *energy* that measures how much it differs from its neighbours, and the seam with the lowest total energy is found with dynamic programming. Repeating "compute energy, find seam, remove seam" k times shrinks the width by k pixels; horizontal seams do the same for the height.

The same mechanism gives several other operations. Duplicating the k lowest-energy seams enlarges an image. Lowering the energy under a user-drawn mask forces seams through an object until it disappears, and raising it under a second mask protects regions such as faces.

In this project we implemented the method from scratch in NumPy, for the course INF605 *Fundamentals of Image Processing*. Our implementation covers:

- the energy map, computed with our own correlation function and five derivative operators;
- the minimum seam with dynamic programming, vertical and horizontal;
- shrinking by seam removal and enlarging by seam insertion;
- object removal and protection with masks;
- comparisons with scaling (`cv2.resize`) and cropping on our own photographs.

OpenCV is used only for file input and output, colour conversion, the baselines and as a reference in our tests.
