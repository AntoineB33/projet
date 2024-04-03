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
    fig = px.scatter_3d(cluster_data, x='x', y='y', z='z', color='cluster')
    filtered_data = cluster_data[cluster_data['cluster'] == selected_cluster]
    fig.add_scatter3d(x=filtered_data['x'], y=filtered_data['y'], z=filtered_data['z'],
                    mode='markers', marker=dict(color='red', size=10),
                    name=f'Cluster {selected_cluster}')
    return fig

@st.cache_data
def plot_metric(df_metric):
    # TODO
    # fig1 = colorize_cluster(df, selected_cluster)
    fig = px.bar(df_metric, x='Descripteur', y='Score AMI', color='Descripteur', barmode='group')
    st.plotly_chart(fig1)

        
# Chargement des données du clustering
df_hist = pd.read_excel("output/save_clustering_hist_kmeans.xlsx")
df_hog = pd.read_excel("output/save_clustering_hog_kmeans.xlsx")
df_metric = pd.read_excel("output/save_metric.xlsx")

if 'Unnamed: 0' in df_metric.columns:
    df_metric.drop(columns="Unnamed: 0", inplace=True)

# Création de deux onglets
tab1, tab2 = st.tabs(["Analyse par descripteur", "Analyse global" ])

# Onglet numéro 1
with tab1:

    st.write('## Résultat de Clustering des données DIGITS')
    st.sidebar.write("####  Veuillez sélectionner les clusters à analyser" )
    # Sélection des descripteurs
    descriptor =  st.sidebar.selectbox('Sélectionner un descripteur', ["HISTOGRAM","HOG"])
    if descriptor=="HISTOGRAM":
        df = df_hist
    if descriptor=="HOG":
        df = df_hog
    # Ajouter un sélecteur pour les clusters
    selected_cluster =  st.sidebar.selectbox('Sélectionner un Cluster', range(10))
    # Filtrer les données en fonction du cluster sélectionné
    cluster_indices = df[df.cluster==selected_cluster].index    
    st.write(f"###  Analyse du descripteur {descriptor}" )
    st.write(f"#### Analyse du cluster : {selected_cluster}")
    st.write(f"####  Visualisation 3D du clustering avec descripteur {descriptor}" )
    # Sélection du cluster choisi
    filtered_data = df[df['cluster'] == selected_cluster]
    # Création d'un graph 3D des clusters
    # TODO : à remplir
    fig1 = colorize_cluster(df, selected_cluster)
    st.plotly_chart(fig)

# Onglet numéro 2
with tab2:
    st.write('## Analyse Global des descripteurs' )
    # Complèter la fonction plot_metric() pour afficher les histogrammes du score AMI
    plot_metric(df_metric)
    st.write('## Métriques ' )
    # TODO :à remplir par un affichage d'un tableau
