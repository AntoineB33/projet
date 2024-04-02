import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import transform
import itertools



def compute_gray_histograms(images):
    """
    Calcule les histogrammes de niveau de gris pour les images MNIST.
    Input : images (list) : liste des images en niveaux de gris
    Output : descriptors (list) : liste des descripteurs d'histogrammes de niveau de gris
    """
    descriptors = []
    for image in images:
        # Convert image to uint8 format (required by cv2.calcHist)
        image_uint8 = image.astype(np.uint8)
        hist = cv2.calcHist([image_uint8], [0], None, [256], [0, 16])
        descriptors.append(hist.flatten())
    return descriptors

def compute_hog_descriptors(images):
    """
    Calcule les descripteurs HOG pour les images en niveaux de gris.
    Input : images (array) : tableau numpy des images
    Output : descriptors (list) : liste des descripteurs HOG
    """
    descriptors = []
    for image in images:
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualize=True)
        descriptors.append(fd)
        
        """fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        
        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')
        
        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        
        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()"""
    
    return descriptors

def compute_sift_descriptors(images):
    """
    Computes SIFT descriptors for a list of images.
    
    :param images: List of images in numpy array format.
    :return: List of SIFT descriptors for each image.
    """
    sift = cv2.SIFT_create()
    descriptors = []
    for image in images:
        # Convert image to uint8 format (required by SIFT)
        image_uint8 = image.astype(np.uint8)
        # Compute keypoints and descriptors
        keypoints, descriptor = sift.detectAndCompute(image_uint8, None)
        descriptors.append(descriptor)
    return descriptors

 
def compute_sift_descriptors2(images):
    img = cv.imread('home.jpg')
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
     
    sift = cv.SIFT_create()
    kp = sift.detect(gray,None)
     
    img=cv.drawKeypoints(gray,kp,img)
     
    cv.imwrite('sift_keypoints.jpg',img)
    return descriptors
