# GET IMAGES
from PIL import Image
import numpy as np
import os

def load_images_from_folder(folder):
    images = []
    labels_true = []
    folder_names = []
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
                    labels_true.append(folder_number)
    return images, labels_true, folder_names


if __name__ == "__main__":
    # affichage d'une image 
    fig = px.imshow(images[0])