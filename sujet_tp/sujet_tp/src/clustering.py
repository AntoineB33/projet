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

class CNNClustering(BaseEstimator, TransformerMixin):
    def __init__(
        self, input_shape, n_clusters=20, learning_rate=0.001, epochs=10, batch_size=32
    ):
        self.input_shape = input_shape  # Dimensions des données d'entrée
        self.n_clusters = n_clusters  # Nombre de clusters à former
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential(
            [
                Conv2D(
                    32,
                    kernel_size=(3, 3),
                    activation="relu",
                    input_shape=self.input_shape,
                ),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(128, activation="relu"),
                Dense(self.n_clusters, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="categorical_crossentropy",  # Vous devrez peut-être ajuster cette partie selon vos données et objectifs
            metrics=["accuracy"],
        )
        return model

    def fit(self, X, y=None):
        # Note: Vous devez transformer vos données en un format compatible avec une CNN dans la méthode fit avant de les passer au modèle.
        # Cela pourrait impliquer le redimensionnement des images pour correspondre à `input_shape` et leur normalisation.
        X_transformed = self._preprocess_data(X)
        # CNNs exigent des étiquettes pour l'apprentissage supervisé, mais vous pouvez utiliser des techniques non supervisées
        # pour générer des étiquettes temporaires si nécessaire ou ajuster cette méthode selon vos besoins.
        self.model.fit(X_transformed, y, epochs=self.epochs, batch_size=self.batch_size)
        return self

    def transform(self, X):
        # Transformer les données pour l'entrée du modèle
        X_transformed = self._preprocess_data(X)
        # Utiliser le modèle pour prédire les "clusters" (sorties du dernier layer avant la couche softmax, par exemple)
        cluster_assignments = self.model.predict(X_transformed)
        return cluster_assignments

    def _preprocess_data(self, X):
        # Mettez ici le code pour prétraiter vos données (redimensionnement, normalisation, etc.)
        return X  # Ceci est un placeholder, remplacez-le par votre propre code de prétraitement

    # Ajoutez toute autre méthode dont vous pourriez avoir besoin pour le traitement des données, l'entraînement, etc.


def cluster_images_with_stacked_rbm(X):
    # Define the Stacked RBM with two layers and respective components
    stacked_rbm = StackedRBM(
        n_components_list=[256, 128], n_iter=20, learning_rate=0.01, batch_size=10
    )

    # Optional: Scale data to [0, 1] - RBMs perform better with binary data or values between 0 and 1
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit the Stacked RBM
    print("Fitting Stacked RBM...")
    stacked_rbm.fit(X_scaled)

    # Transform the data to the new feature space
    X_transformed = stacked_rbm.transform(X_scaled)

    # Cluster the data using KMeans
    print("Clustering with KMeans...")
    kmeans = KMeans(n_clusters=20, random_state=42)
    kmeans.fit(X_transformed)

    # Return the cluster labels
    return kmeans.labels_


# Supposons que X soit votre matrice de données d'images prétraitées.
# Vous pouvez appeler la fonction cluster_images_with_stacked_rbm(X) pour effectuer le clustering.

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

    # Affichons les résultats
    if bool_show:
        print(f"########## Métrique descripteur : {name_descriptor}")
        print(f"Adjusted Mutual Information: {ami}")
        # Calcul du Silhouette Score
        score = silhouette_score(descriptors, labels_pred)
        print("Silhouette Score:", score)
    if bool_return:
        return {"ami": ami, "name_model": name_model}
