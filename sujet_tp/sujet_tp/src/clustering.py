from sklearn.metrics import adjusted_mutual_info_score
import numpy as np

from sklearn.neural_network import BernoulliRBM
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Input

eyyy = 1

class StackedRBM(TransformerMixin):
    def __init__(
        self, n_components_list=[256, 128], n_iter=10, learning_rate=0.01, batch_size=10
    ):
        self.rbms = []
        self.n_components_list = n_components_list
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        for n_components in self.n_components_list:
            rbm = BernoulliRBM(
                n_components=n_components,
                n_iter=self.n_iter,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                verbose=True,
            )
            self.rbms.append(rbm)

    def fit(self, X, y=None):
        input_data = X
        for rbm in self.rbms:
            rbm.fit(input_data)
            input_data = rbm.transform(input_data)
        return self

    def transform(self, X):
        input_data = X
        for rbm in self.rbms:
            input_data = rbm.transform(input_data)
        return input_data

from sklearn.metrics import silhouette_score

def show_metric(
    labels_true,
    labels_pred,
    descriptors,
    bool_return=False,
    name_descriptor="",
    name_model="kmeans",
    bool_show=True,
):
    """
    Fonction d'affichage et création des métrique pour le clustering.
    Input :
    - labels_true : étiquettes réelles des données
    - labels_pred : étiquettes prédites des données
    - descriptors : ensemble de descripteurs utilisé pour le clustering
    - bool_return : booléen indiquant si les métriques doivent être retournées ou affichées
    - name_descriptor : nom de l'ensemble de descripteurs utilisé pour le clustering
    - name_model : nom du modèle de clustering utilisé
    - bool_show : booléen indiquant si les métriques doivent être affichées ou non

    Output :
    - dictionnaire contenant les métriques d'évaluation des clusters
    """

    ami = adjusted_mutual_info_score(labels_true, labels_pred)
    silouhette = silhouette_score(descriptors, labels_pred)

    # Affichons les résultats
    if bool_show:
        print(f"########## Métrique descripteur : {name_descriptor}")
        print(f"Adjusted Mutual Information: {ami}")
        # Calcul du Silhouette Score
        print("Silhouette Score:", silouhette)
    if bool_return:
        return {"ami": ami, "silhouette": silouhette, "descriptor": name_descriptor, "name_model": name_model}
