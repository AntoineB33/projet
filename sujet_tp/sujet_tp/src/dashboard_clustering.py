import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px
import cv2
from images import load_images_from_folder
from constant import  PATH_OUTPUT, MODEL_CLUSTERING, PATH_DATA, PATH_DATA_ALL

@st.cache_data
def colorize_cluster(cluster_data, selected_cluster):
    fig = px.scatter_3d(cluster_data, x="x", y="y", z="z", color="cluster")
    filtered_data = cluster_data[cluster_data["cluster"] == selected_cluster]
    fig.add_scatter3d(
        x=filtered_data["x"],
        y=filtered_data["y"],
        z=filtered_data["z"],
        mode="markers",
        marker=dict(color="red", size=10),
        name=f"Cluster {selected_cluster}",
    )
    return fig

import plotly.graph_objects as go
@st.cache_data
def plot_metric(df_metric):
    # Préparation des données pour le graphique
    metrics = ['ami', 'silhouette']
    models = df_metric['name_model'].unique()
    figures = []

    # Génération d'un graphique pour chaque métrique
    for metric in metrics:
        fig = go.Figure()
        for model in models:
            df_filtered = df_metric[df_metric['name_model'] == model]
            fig.add_trace(go.Bar(
                x=df_filtered['descriptor'],
                y=df_filtered[metric],
                name=model
            ))
        
        # Configuration du graphique
        fig.update_layout(
            title=f"{metric.upper()} by Model and Descriptor",
            xaxis_title="Descriptor",
            yaxis_title=metric.upper(),
            barmode='group'
        )
        figures.append(fig)
    
    # Affichage des graphiques
    for fig in figures:
        st.plotly_chart(fig)
    # TODO
    # fig1 = colorize_cluster(df, selected_cluster)
    # fig1 = px.bar(
    #     df_metric, x="Descripteur", y="ami", color="Descripteur", barmode="group"
    # )
    # fig2 = px.bar(
    #     df_metric,
    #     x="name_model",
    #     y="silhouette",
    #     color="Descripteur",
    #     barmode="group",
    # )
    # st.plotly_chart(fig1)
    # st.plotly_chart(fig2)


# Étape 3: Sélectionner et afficher une image d'un cluster choisi
def display_image_from_cluster(df, cluster_indices, images):
    if not cluster_indices.empty:
        # Sélectionnez la première image du cluster pour l'affichage, par exemple
        img = images[cluster_indices[0]]
        st.image(img, caption=f"Image from selected cluster.")
    else:
        st.write(f"No images found for selected cluster.")

# Chargement des données du clustering
# descriptors0 = ["RGB", "HSV"]
# descriptors = ["HISTOGRAM", "HOG", "SIFT"]
modeles_title = ["Stacked RBM Kmeans","XGBoost"]
modeles = ["rbm_Kmeans", "XGBoost"]
descs0 = ["", "HSV"]
descs = ["hist", "hog", "sift"]
df_list = []
for d0 in descs0:
    for d in descs:
        for m in modeles:
            df_list.append(pd.read_excel(f"output/save_clustering_{d0}_{d}_{m}.xlsx"))
df_hsv_hist_rbm = pd.read_excel("output/save_clustering_hsv_hist_rbm_kmeans.xlsx")
# df_hsv_hog_rbm = pd.read_excel("output/save_clustering_hsv_hog_rbm_kmeans.xlsx")
# df_hsv_sift_rbm = pd.read_excel("output/save_clustering_hsv_sift_rbm_kmeans.xlsx")
# df_hist_rbm = pd.read_excel("output/save_clustering_hist_rbm_kmeans.xlsx")
# df_hog_rbm = pd.read_excel("output/save_clustering_hog_rbm_kmeans.xlsx")
# df_sift_rbm = pd.read_excel("output/save_clustering_sift_rbm_kmeans.xlsx")
df_metric = pd.read_excel("output/save_metric.xlsx")
folder_path = PATH_DATA_ALL + "/code_test"
images, labels_true, folder_names = load_images_from_folder(folder_path)

if "Unnamed: 0" in df_metric.columns:
    df_metric.drop(columns="Unnamed: 0", inplace=True)

# Création de deux onglets
tab1, tab2 = st.tabs(["Analyse par descripteur", "Analyse global"])


descsTitle0 = ["RGB", "HSV"]
descsTitle = ["HISTOGRAM", "HOG", "SIFT", "aucun"]

# Onglet numéro 1
with tab1:

    st.write("## Résultat de Clustering des données DIGITS")
    st.sidebar.write("####  Veuillez sélectionner les clusters à analyser")
    # Sélection des descripteurs
    descriptor0 = st.sidebar.selectbox("Sélectionner un descripteur", descsTitle0)
    descriptor = st.sidebar.selectbox(
        "Sélectionner un descripteur", descsTitle
    )
    model= st.sidebar.selectbox("Sélectionner un modèle", modeles_title)
    df = df_list[descsTitle0.index(descriptor0) * len(descriptor0) + descsTitle.index(descriptor) * len(descriptor) + modeles_title.index(model)]
    # Ajouter un sélecteur pour les clusters
    selected_cluster = st.sidebar.selectbox("Sélectionner un Cluster", range(20))
    # Filtrer les données en fonction du cluster sélectionné
    cluster_indices = df[df.cluster == selected_cluster].index
    st.write(f"###  Analyse du descripteur {descriptor}")
    st.write(f"#### Analyse du cluster : {selected_cluster}")
    st.write(f"####  Visualisation 3D du clustering avec descripteur {descriptor}")
    # Sélection du cluster choisi
    filtered_data = df[df["cluster"] == selected_cluster]
    # Création d'un graph 3D des clusters
    # TODO : à remplir
    fig1 = colorize_cluster(df, selected_cluster)
    st.plotly_chart(fig1)
    display_image_from_cluster(df, cluster_indices, images)

# Onglet numéro 2
with tab2:
    st.write("## Analyse Global des descripteurs")
    # Complèter la fonction plot_metric() pour afficher les histogrammes du score AMI
    plot_metric(df_metric)
    st.write("## Métriques ")
    # TODO :à remplir par un affichage d'un tableau
    st.dataframe(df_metric)
