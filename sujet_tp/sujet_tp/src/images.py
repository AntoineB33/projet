# GET IMAGES
from PIL import Image
import numpy as np
import os

def load_images_from_folder(folder):
    images = []
    labels_true = []
    folder_names = []
    smallest_width = float('inf')
    smallest_height = float('inf')
    smallest_width2 = 0
    smallest_height2 = 0
    for folder_number, sub_folder in enumerate(sorted(os.listdir(folder))):
        sub_folder_path = os.path.join(folder, sub_folder)
        if os.path.isdir(sub_folder_path):
            folder_names.append(sub_folder)
            for filename in sorted(os.listdir(sub_folder_path)):
                img_path = os.path.join(sub_folder_path, filename)
                if os.path.isfile(img_path):
                    img = Image.open(img_path)
                    img_matrix = np.array(img)
                    images.append(img_matrix[:256,:256])
                    # images.append(img_matrix)
                    labels_true.append(folder_number)
                    # height, width = img_matrix.shape[:2]
                    # smallest_width = min(smallest_width, width)
                    # smallest_height = min(smallest_height, height)
                    # smallest_width2 = max(smallest_width2, width)
                    # smallest_height2 = max(smallest_height2, height)
    return images, labels_true, folder_names, smallest_height, smallest_width, smallest_height2, smallest_width2


if __name__ == "__main__":
    # affichage d'une image 
    fig = px.imshow(images[0])