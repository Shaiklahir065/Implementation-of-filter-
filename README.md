# Implementation of filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step 1:
Import necessary libraries: OpenCV, NumPy, and Matplotlib.Read an image, convert it to RGB format, define an 11x11 averaging kernel, and apply 2D convolution filtering.Display the original and filtered images side by side using Matplotlib.

### Step 2:
Define a weighted averaging kernel (kernel2) and apply 2D convolution filtering to the RGB image (image2).Display the resulting filtered image (image4) titled 'Weighted Averaging Filtered' using Matplotlib's imshow function.

### Step 3:

Apply Gaussian blur with a kernel size of 11x11 and standard deviation of 0 to the RGB image (image2).Display the resulting Gaussian-blurred image (gaussian_blur) titled 'Gaussian Blurring Filtered' using Matplotlib's imshow function.
### Step 4:
Apply median blur with a kernel size of 11x11 to the RGB image (image2).Display the resulting median-blurred image (median) titled 'Median Blurring Filtered' using Matplotlib's imshow function.

### Step 5 :
Define a Laplacian kernel (kernel3) and perform 2D convolution filtering on the RGB image (image2).Display the resulting filtered image (image5) titled 'Laplacian Kernel' using Matplotlib's imshow function.
### Step 6 :
Apply the Laplacian operator to the RGB image (image2) using OpenCV's cv2.Laplacian function.Display the resulting image (new_image) titled 'Laplacian Operator' using Matplotlib's imshow function.

## Program:

 ### Developed By:Shaik Lahir
 ### Register Number:212224240148

### 1. Smoothing Filters

#### i) Original image
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
image1 = cv2.imread('golden retriver.jpg')
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title('Original')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/65c4ba00-3ac4-4d4c-aa10-c26007c9956d)

#### ii) Using Averaging Filter
```
kernel = np.ones((11,11), np. float32)/121
image3 = cv2.filter2D(image2, -1, kernel)

plt.subplot(1,2,2)
plt.imshow(image3)
plt.title('Filtered')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/4ecfeb2d-24fe-4b33-a898-b0f420e8d1d6)


#### ii) Using Weighted Averaging Filter
```python
image1 = cv2.imread('golden retriver.jpg')
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

kernel2 = np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image4 = cv2.filter2D(image2, -1, kernel2)
plt.imshow(image4)
plt.title('Weighted Averaging Filtered')
```
![image](https://github.com/user-attachments/assets/01185d6e-7fb6-410c-a8cd-e15310fca0c1)

#### iii) Using Gaussian Filter
```python
gaussian_blur = cv2.GaussianBlur(src=image2, ksize=(11,11), sigmaX=0, sigmaY=0)
plt.imshow(gaussian_blur)
plt.title(' Gaussian Blurring Filtered')
```
![image](https://github.com/user-attachments/assets/1c1f25af-fbac-40ee-bc12-83028d8e95fa)

#### iv) Using Median Filter
```python
median=cv2.medianBlur (src=image2, ksize=11)
plt.imshow(median)
plt.title(' Median Blurring Filtered')
```
![image](https://github.com/user-attachments/assets/3a59d9c1-5965-419d-a094-90b86dda3567)

### 2. Sharpening Filters
#### i) Using Laplacian Kernel
```python
kernel3 = np.array([[0,1,0], [1, -4,1],[0,1,0]])
image5 =cv2.filter2D(image2, -1, kernel3)
plt.imshow(image5)
plt.title('Laplacian Kernel')
```
![image](https://github.com/user-attachments/assets/48b14cae-ca17-432b-8d79-d663352acad4)

#### ii) Using Laplacian Operator
```python
new_image = cv2.Laplacian (image2, cv2.CV_64F)
plt.imshow(new_image)
plt.title('Laplacian Operator')
```
![image](https://github.com/user-attachments/assets/9e1e2e85-7011-4ab2-bb56-68a4d53047ea)

## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
