from sklearn.preprocessing import StandardScaler
import os
import pandas as pd

from sujet_tp.sujet_tp.src.features import *
from clustering import *
from utils import *
from constant import PATH_OUTPUT, MODEL_CLUSTERING

from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
from sklearn import datasets
import plotly as plt
import plotly.express as px
from features import *
from clustering import *
from utils import *
from constant import  PATH_OUTPUT, MODEL_CLUSTERING, PATH_DATA, PATH_DATA_ALL

from sklearn.cluster import KMeans

from sklearn.pipeline import Pipeline

from skimage.transform import resize

from sklearn.preprocessing import StandardScaler
from images import load_images_from_folder

import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import skimage as skimage
%matplotlib inline
import xgboost as xgb
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.metrics import accuracy_score, silhouette_score, adjusted_mutual_info_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from sklearn.neural_network import BernoulliRBM


params = {
    'objective': 'multi:softmax', 
    'num_class': len(np.unique(labels_encoded)),
    'eta': 0.1,  # Decrease learning rate [0.01-0.3] Lower values make the model more robust but require more boosting rounds.
    'max_depth': 8,  # Adjust max depth of trees [3-10] Higher values allow the model to capture more complex interactions in the data but can lead to overfitting.
    'num_boost_round': 100,  # Increase number of boosting rounds [50-500]
    'subsample': 0.75,  # Reduce subsample ratio [0.5-1] Lower values introduce more randomness and can prevent overfitting.
    'colsample_bytree': 0.85,  # Reduce column subsample ratio [0.5-1] Lower values introduce more randomness in feature selection.
    'lambda': 10,  # L2 regularization term [0-10+] Higher values add more regularization to prevent overfitting.
    'alpha': 10  # L1 regularization term [0-10+] Higher values add more regularization to prevent overfitting.
}

descsTitle0 = ["RGB", "HSV"]
descsTitle = ["HISTOGRAM", "HOG", "SIFT"]
modeles_title = ["Stacked RBM Kmeans","XGBoost"]

def pipeline():

    # Example usage:
    folder_path = PATH_DATA_ALL + "/code_test"
    images, labels_true, folder_names = load_images_from_folder(folder_path)
    taille = len(images)
    nombre_de_canaux = 3
   
    print("\n\n ##### Extraction de Features ######")
    

    images_to_use = [images, convert_color_space(images, "HSV")]
    descs0 = ["", "HSV"]
    descs = ["hist", "hog", "sift"]
    list_dict=[]
    
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels_true)

    for d0 in range(len(descs0)):
        descriptors_hog = compute_hog_descriptors(images_to_use[d0])
        descriptors_hist = compute_gray_histograms(images_to_use[d0])
        sift_descriptors = extract_sift_features(images_to_use[d0])
        descriptors_sift = create_bag_of_features(sift_descriptors, n_clusters=20)
        descriptors = [descriptors_hist, descriptors_hog, descriptors_sift]
        for d in range(len(descs)):

            # Présumons que `images` est votre liste d'images prétraitées et aplatie en vecteurs

            # Initialisation de la classe StackedRBM
            stacked_rbm = StackedRBM(n_components_list=[256, 128], n_iter=10, learning_rate=0.01, batch_size=10)

            # Ajustement des RBMs sur les données d'image
            stacked_rbm.fit(descriptors[d])

            # Transformation des images en nouvelles représentations avec les RBMs entraînés
            transformed_images = stacked_rbm.transform(descriptors[d])

            # Normalisation des caractéristiques pour améliorer les performances de K-Means
            scaler = StandardScaler()
            transformed_images_scaled = conversion_3d(transformed_images)

            # Clustering avec K-Means
            kmeans = KMeans(n_clusters=20, random_state=42)
            clusters = kmeans.fit_predict(transformed_images_scaled)
            
            
            metric = show_metric(labels_true, clusters, transformed_images_scaled, bool_show=True, name_descriptor=f"{descsTitle0[d0]} et {descsTitle[d]}", name_model = "Stacked RBM et Kmeans", bool_return=True)
            list_dict.append(metric)
            
            
            scaler = StandardScaler()
            descriptors_norm = scaler.fit_transform(descriptors[d])
            
            x_3d_norm = conversion_3d(descriptors_norm)
            
            df = create_df_to_export(x_3d_norm, labels_true, kmeans.labels_)

            # sauvegarde des données
            df.to_excel(PATH_OUTPUT+f"/save_clustering_{descs0[d0]}_{descs[d]}_rbm_kmeans.xlsx")
            print(f"save_clustering_{descs0[d0]}_{descs[d]}_rbm_kmeans.xlsx")
            
            
            
            X_train, X_test, y_train, y_test = train_test_split(descriptors[d], labels_encoded, test_size=0.2, random_state=42)
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_train, label=y_test)
            bst = xgb.train(params, dtrain, num_boost_round=params['num_boost_round'])
            preds = bst.predict(dtest)
            metric = show_metric(labels_true, clusters, transformed_images_scaled, bool_show=True, name_descriptor=f"{descsTitle0[d0]} et {descsTitle[d]}", name_model = "XGBoost", bool_return=True)
            list_dict.append(metric)
            
            
            scaler = StandardScaler()
            descriptors_norm = scaler.fit_transform(descriptors[d])
            
            x_3d_norm = conversion_3d(descriptors_norm)
            
            df = create_df_to_export(x_3d_norm, labels_true, preds)

            # sauvegarde des données
            df.to_excel(PATH_OUTPUT+f"/save_clustering_{descs0[d0]}_{descs[d]}_rbm_kmeans.xlsx")
            print(f"save_clustering_{descs0[d0]}_{descs[d]}_rbm_kmeans.xlsx")
            
            
            
            
            bandwidth = estimate_bandwidth(X_train_normalized, quantile=0.06)
            mean_shift = MeanShift(bandwidth=bandwidth,cluster_all=True)
            mean_shift.fit(X_train_normalized)
            
    df_metric = pd.DataFrame(list_dict)
    df_metric.to_excel(PATH_OUTPUT+"/save_metric.xlsx")
    
    

    print("- calcul features hog...")
    # TODO
    print("- calcul features Histogram...")
    # TODO
    print("- calcul features sift...")
    # TODO
    print("- calcul features hsv...")
    # TODO


    print("\n\n ##### Clustering ######")
    number_cluster = 20

    print("-- Mean Shift --")
    # TODO
    print("- calcul Mean Shift avec features HOG ...")
    # TODO
    print("- calcul Mean Shift avec features Histogram...")
    # TODO
    print("- calcul Mean Shift avec features SIFT...")
    # TODO
    print("- calcul Mean Shift avec features HSV ...")
    # TODO

    print("-- Stacked RMB (Restricted Boltzmann Machine) --")
    # TODO
    print("- calcul Stacked RMB avec features HOG ...")
    # TODO
    print("- calcul Stacked RMB avec features Histogram...")
    # TODO
    print("- calcul Stacked RMB avec features SIFT ...")
    # TODO
    print("- calcul Stacked RMB avec features HSV ...")
    # TODO

    print("-- XGBoost (Extrem Gradien boosting) --")
    # TODO
    print("- calcul XGBoost avec features HOG ...")
    # TODO
    print("- calcul XGBoost avec features Histogram...")
    # TODO
    print("- calcul XGBoost avec features SIFT ...")
    # TODO
    print("- calcul XGBoost avec features HSV ...")
    # TODO

    print("-- RCA (Random Clustering Asign) --")
    # TODO
    print("- calcul RCA avec features HOG ...")
    # TODO
    print("- calcul RCA avec features Histogram...")
    # TODO
    print("- calcul RCA avec features SIFT ...")
    # TODO
    print("- calcul RCA avec features HSV ...")
    # TODO
    

    # print("\n\n ##### Résultat ######")
    # metric_hist = show_metric(labels_true, kmeans_hist.labels_, descriptors_hist, bool_show=True, name_descriptor="HISTOGRAM", bool_return=True)
    # metric_hog = show_metric(labels_true, kmeans_hog.labels_, descriptors_hog,bool_show=True, name_descriptor="HOG", bool_return=True)

    # print("- export des données vers le dashboard")
    # # conversion des données vers le format du dashboard
    # list_dict = [metric_hist, metric_hog]
    # df_metric = pd.DataFrame(list_dict)
    
    # # Normalisation des données
    # scaler = StandardScaler()
    # descriptors_hist_norm = scaler.fit_transform(descriptors_hist)
    # descriptors_hog_norm = scaler.fit_transform(descriptors_hog)

    # #conversion vers un format 3D pour la visualisation
    # print("- conversion vers le format 3D ...")
    # x_3d_hist = conversion_3d(descriptors_hist_norm)
    # x_3d_hog = conversion_3d(descriptors_hog_norm)

    # # création des dataframe pour la sauvegarde des données pour la visualisation
    # df_hist = create_df_to_export(x_3d_hist, labels_true, kmeans_hist.labels_)
    # df_hog = create_df_to_export(x_3d_hog, labels_true, kmeans_hog.labels_)

    # # Vérifie si le dossier existe déjà
    # if not os.path.exists(PATH_OUTPUT):
    #     # Crée le dossier
    #     os.makedirs(PATH_OUTPUT)

    # # sauvegarde des données
    # print("- enregistrement des fichiers ...")
    # df_hist.to_excel(PATH_OUTPUT+"/save_clustering_hist_kmeans.xlsx")
    # df_hog.to_excel(PATH_OUTPUT+"/save_clustering_hog_kmeans.xlsx")
    # df_metric.to_excel(PATH_OUTPUT+"/save_metric.xlsx")
    print("Fin. \n\n Pour avoir la visualisation dashboard, veuillez lancer la commande : streamlit run dashboard_clustering.py")


if __name__ == "__main__":
    pipeline()