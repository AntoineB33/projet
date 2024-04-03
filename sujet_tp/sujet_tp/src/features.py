import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import transform
import itertools

from skimage.color import rgb2gray

# SIFT
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage.feature import SIFT
import numpy as np


# Fonction pour convertir une liste d'images RGB en un espace colorimétrique différent
def convert_color_space(images, target_space):
    converted_images = []
    for image in images:
        if target_space == "HSV":
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif target_space == "Lab":
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        else:
            raise ValueError("Unsupported color space")
        converted_images.append(converted_image)
    return converted_images


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
        fd, hog_image = hog(
            image_gray,
            orientations=8,
            pixels_per_cell=(8, 8),
            cells_per_block=(1, 1),
            visualize=True,
        )
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


def convert_to_grayscale(image):
    """Converts an RGB image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

from skimage.color import rgb2gray
from skimage.feature import SIFT
import numpy as np

# Étape 1: Extraction des caractéristiques SIFT avec conversion en niveaux de gris pour images RGB
def extract_sift_features(images):
    sift_descriptors = []
    sift = SIFT()
    for image in images:
        # Convertir l'image RGB en niveaux de gris
        gray_image = rgb2gray(image)
        sift.detect_and_extract(gray_image)
        descriptors = sift.descriptors
        sift_descriptors.append(descriptors)
    return sift_descriptors

# # Étape 1: Extraction des caractéristiques SIFT
# def extract_sift_features(images):
#     sift_descriptors = []
#     sift = SIFT()
#     for image in images:
#         gray_image = convert_to_grayscale(image)  # Convert image to grayscale
#         keypoints, descriptors = sift.detectAndCompute(
#             gray_image, None
#         )  # Adjusted to use detectAndCompute
#         sift_descriptors.append(descriptors)
#     return sift_descriptors


# Étape 2: Aggregation des descripteurs SIFT (exemple simple avec K-Means pour créer un "Bag of Features")
def create_bag_of_features(sift_descriptors, n_clusters=20):
    all_descriptors = np.vstack(sift_descriptors)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(all_descriptors)
    features = np.array(
        [kmeans.predict(descriptors) for descriptors in sift_descriptors]
    )
    # Création d'un histogramme de caractéristiques pour chaque image
    histograms = np.array(
        [np.bincount(feature, minlength=n_clusters) for feature in features]
    )
    return histograms


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
