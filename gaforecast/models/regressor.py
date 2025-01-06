import logging
from typing import Tuple, Union

import numpy as np
import pandas as pd  # type: ignore
from pyts.image import GramianAngularField  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from tensorflow.keras.applications import ResNet50  # type: ignore
from tensorflow.keras.layers import Dense, Flatten  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.optimizers import SGD  # type: ignore
from tensorflow.keras.utils import Sequence  # type: ignore
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from openTSNE import TSNE
from sklearn.preprocessing import MinMaxScaler


def find_optimal_clusters_silhouette(X, max_clusters=30):
    best_k = 2
    best_score = -1
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        score = silhouette_score(X, cluster_labels)
        print(f"Clusters: {k}, Silhouette Score: {score}")
        if score > best_score:
            best_k = k
            best_score = score
    print(f"Optimal number of clusters based on silhouette score: {best_k}")
    return best_k


def balance_data_with_tsne_auto_clusters(X:pd.DataFrame, y, random_state=None, max_clusters=30):
    # Step 1: Reduce dimensionality with openTSNE
    print("Reducing dimensionality using openTSNE...")
    tsne = TSNE(n_components=2, random_state=random_state)
    X_embedded = tsne.fit(X.to_numpy())  # openTSNE's fit method transforms the data

    # Step 2: Find optimal clusters
    print("Finding optimal number of clusters...")
    optimal_cluster_count = find_optimal_clusters_silhouette(X_embedded, max_clusters=max_clusters)
    print(f"Optimal Clusters: {optimal_cluster_count}")

    # Step 3: Cluster the data
    print("Clustering data with KMeans...")
    kmeans = KMeans(n_clusters=optimal_cluster_count, random_state=random_state)
    cluster_labels = kmeans.fit_predict(X_embedded)

    # Step 4: Balance the data
    print("Balancing the dataset...")
    balanced_indices = []
    for cluster in np.unique(cluster_labels):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        min_cluster_size = min(len(cluster_indices), len(X) // optimal_cluster_count)
        sampled_indices = np.random.choice(cluster_indices, size=min_cluster_size, replace=False)
        balanced_indices.extend(sampled_indices)

    balanced_X = X.iloc[balanced_indices]
    balanced_y = y.iloc[balanced_indices]

    return balanced_X, balanced_y

class GAForecastRegressor(BaseEstimator):
    def __init__(self, batch_size: int = 128, epochs: int = 10, max_clusters: int = 30):
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = "mean_squared_error"  # Regression loss
        self.max_clusters = max_clusters
        self.scaler = MinMaxScaler(feature_range=(-1, 1), clip=True)

    def fit(
            self,
            X: Union[np.ndarray, pd.DataFrame],
            Y: Union[np.ndarray, pd.DataFrame] = None,
    ) -> None:
        X, Y = self._balance_data(X, Y)
        X = self._normalize_data(X, fit=True)
        image_size = self._get_image_size(X)

        data_generator = GAFDataGenerator(X, Y, self.batch_size, image_size)
        self.model = self._create_model(image_size)

        # Define EarlyStopping callback
        early_stopping = EarlyStopping(
            monitor="loss",  # Metric to monitor
            patience=8,  # Number of epochs with no improvement before stopping
            verbose=1,  # Verbosity level: 1 = display messages
            restore_best_weights=True  # Restore the weights from the best epoch
        )

        self.model.fit(
            data_generator,
            epochs=self.epochs,
            steps_per_epoch=len(data_generator),
            shuffle=True,
            callbacks=[early_stopping],
        )
        return None

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        X = self._normalize_data(X, fit=False)
        data_generator = GAFDataGenerator(
            X, np.zeros(len(X)), self.batch_size, self._get_image_size(X)
        )
        predictions = self.model.predict(data_generator)
        return predictions  # Direct numerical predictions for regression

    def _normalize_data(self, X: Union[np.ndarray, pd.DataFrame], fit: bool) -> Union[np.ndarray, pd.DataFrame]:
        if fit:
            # Fit the scaler and transform the data
            if isinstance(X, pd.DataFrame):
                X_normalized = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)
            else:
                X_normalized = self.scaler.fit_transform(X)
        else:
            # Only transform the data without fitting
            if isinstance(X, pd.DataFrame):
                X_normalized = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
            else:
                X_normalized = self.scaler.transform(X)
        return X_normalized

    def _balance_data(
            self,
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.DataFrame],
    ) -> Tuple[np.ndarray, np.ndarray]:
        return balance_data_with_tsne_auto_clusters(X, y, max_clusters=self.max_clusters)

    def _closest_smaller_divisible_by_3(self, number: int) -> int:
        remainder = number % 3
        return number - remainder

    def _get_image_size(self, data: np.ndarray) -> int:
        return self._closest_smaller_divisible_by_3(data.shape[1])

    def _create_model(self, image_size: int) -> Model:
        input_shape = (image_size, image_size, 3)

        base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
        for layer in base_model.layers:
            layer.trainable = True

        x = base_model.output
        x = Flatten()(x)
        x = Dense(image_size, activation=LeakyReLU(alpha=0.1))(x)
        x = Dense(image_size, activation=LeakyReLU(alpha=0.1))(x)

        output = Dense(1, activation="linear")(x)  # Single neuron, linear activation
        from tensorflow.keras.optimizers import Adam
        model = Model(inputs=base_model.input, outputs=output)
        model.compile(
            optimizer=SGD(learning_rate=0.001, momentum=0.9, clipnorm=1.0),
            loss=self.loss,
            metrics=["mean_squared_error"],  # Useful regression metric
        )
        return model


class GAFDataGenerator(Sequence):
    def __init__(
            self,
            data: Union[np.ndarray, pd.DataFrame],
            labels: Union[np.ndarray, pd.DataFrame],
            batch_size: int,
            image_size: int,
    ):
        self.data = np.array(data)
        self.labels = np.array(labels).reshape(-1, 1)  # Ensure labels are reshaped
        self.batch_size = batch_size
        self.image_size = image_size
        self.gaf = GramianAngularField(image_size=image_size, sample_range=None)

    def __len__(self) -> int:
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_data = self.data[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]

        transformed_data = []
        for row in batch_data:
            row = row.copy()
            row_data = np.array(row).reshape(1, -1)
            gaf_image = self.gaf.fit_transform(row_data)
            gaf_image = gaf_image.reshape(self.image_size, self.image_size, 1)
            gaf_image = np.repeat(gaf_image, 3, axis=-1)
            transformed_data.append(gaf_image)

        return np.array(transformed_data), np.array(batch_labels)
