import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px
import cv2


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


# Chargement des données du clustering
df_hsv_hist_rbm = pd.read_excel("output/save_clustering_hsv_hist_rbm_kmeans.xlsx")
# df_hsv_hog_rbm = pd.read_excel("output/save_clustering_hsv_hog_rbm_kmeans.xlsx")
# df_hsv_sift_rbm = pd.read_excel("output/save_clustering_hsv_sift_rbm_kmeans.xlsx")
# df_hist_rbm = pd.read_excel("output/save_clustering_hist_rbm_kmeans.xlsx")
# df_hog_rbm = pd.read_excel("output/save_clustering_hog_rbm_kmeans.xlsx")
# df_sift_rbm = pd.read_excel("output/save_clustering_sift_rbm_kmeans.xlsx")
df_metric = pd.read_excel("output/save_metric.xlsx")

if "Unnamed: 0" in df_metric.columns:
    df_metric.drop(columns="Unnamed: 0", inplace=True)

# Création de deux onglets
tab1, tab2 = st.tabs(["Analyse par descripteur", "Analyse global"])

# Onglet numéro 1
with tab1:

    st.write("## Résultat de Clustering des données DIGITS")
    st.sidebar.write("####  Veuillez sélectionner les clusters à analyser")
    # Sélection des descripteurs
    descriptor0 = st.sidebar.selectbox("Sélectionner un descripteur", ["RGB", "HSV"])
    descriptor = st.sidebar.selectbox(
        "Sélectionner un descripteur", ["HISTOGRAM", "HOG", "SIFT"]
    )
    if descriptor0 == "HSV":
        if descriptor == "HISTOGRAM":
            df = df_hsv_hist_rbm
        elif descriptor == "HOG":
            df = df_hsv_hist_rbm
        else:
            df = df_hsv_hist_rbm
    else:
        if descriptor == "HISTOGRAM":
            df = df_hsv_hist_rbm
        elif descriptor == "HOG":
            df = df_hsv_hist_rbm
        else:
            df = df_hsv_hist_rbm
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

# Onglet numéro 2
with tab2:
    st.write("## Analyse Global des descripteurs")
    # Complèter la fonction plot_metric() pour afficher les histogrammes du score AMI
    plot_metric(df_metric)
    st.write("## Métriques ")
    # TODO :à remplir par un affichage d'un tableau
