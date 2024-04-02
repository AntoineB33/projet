import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import transform
import itertools

from skimage.color import rgb2gray


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
    # descriptors = []
    # for image in images:
    #     fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
    #                 cells_per_block=(1, 1), visualize=True)
    #     descriptors.append(fd)

    # return descriptors
    descriptors = []
    for image in images:
        # Vérifie si l'image est en couleur (vérifie si elle a 3 dimensions)
        if image.ndim == 3:
            image_gray = rgb2gray(image)
        else:
            image_gray = image

        # Calcul du HOG sans spécifier channel_axis pour les images en niveaux de gris
        fd, hog_image = hog(image_gray, orientations=8, pixels_per_cell=(8, 8),
                            cells_per_block=(1, 1), visualize=True)
        # # Calcul du HOG
        # fd, hog_image = hog(
        #     image_gray,
        #     orientations=8,
        #     pixels_per_cell=(8, 8),
        #     cells_per_block=(1, 1),
        #     visualize=True,
        #     channel_axis=-1,
        # )
        descriptors.append(fd)
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
    img = cv.imread("home.jpg")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp = sift.detect(gray, None)

    img = cv.drawKeypoints(gray, kp, img)

    cv.imwrite("sift_keypoints.jpg", img)
    return descriptors
