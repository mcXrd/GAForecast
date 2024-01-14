import logging
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd  # type: ignore
from pyts.image import GramianAngularField  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from tensorflow.keras.applications import ResNet50  # type: ignore
from tensorflow.keras.layers import Dense, Flatten  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.optimizers import SGD  # type: ignore
from tensorflow.keras.utils import Sequence  # type: ignore

logger = logging.getLogger()


class GAForecastBinaryClassifier(BaseEstimator):
    def __init__(self, batch_size: int = 128, epochs: int = 10):
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = 2
        self.loss = "kullback_leibler_divergence"

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame] = None,
    ) -> None:
        assert set(y) == {0, 1}, "Labels can have only values 0 or 1!"
        X, y = self._balance_data(X, y)
        image_size = self._get_image_size(X)

        data_generator = GAFDataGenerator(X, y, self.batch_size, image_size)
        self.model = self._create_model(image_size)
        self.model.fit(
            data_generator,
            epochs=self.epochs,
            steps_per_epoch=len(data_generator),
            shuffle=True,
        )
        return None

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        data_generator = GAFDataGenerator(
            X, np.zeros(len(X)), self.batch_size, self._get_image_size(X)
        )
        predictions = self.model.predict(data_generator)
        predictions = self._make_predictions_binary(predictions)
        return predictions

    def _balance_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame],
        random_state: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Combine the data and labels into a single DataFrame
        df = pd.DataFrame(X)
        df["label"] = y

        # Separate the dataset into two subsets based on the class label
        class_0_subset = df[df["label"] == 0]
        class_1_subset = df[df["label"] == 1]

        # Determine the size of the minority class
        minority_class_size = min(len(class_0_subset), len(class_1_subset))

        # Randomly sample from the majority class to match the minority class size
        balanced_class_0_subset = class_0_subset.sample(
            n=minority_class_size, random_state=random_state
        )
        balanced_class_1_subset = class_1_subset.sample(
            n=minority_class_size, random_state=random_state
        )

        # Calculate how many samples were dropped from the majority class
        samples_dropped = (
            max(len(class_0_subset), len(class_1_subset)) - minority_class_size
        )

        # Combine the balanced subsets to create the balanced dataset
        balanced_dataset = pd.concat([balanced_class_0_subset, balanced_class_1_subset])

        # Shuffle the balanced dataset to ensure randomness
        balanced_dataset = balanced_dataset.sample(frac=1, random_state=random_state)

        logger.info(
            f"{samples_dropped} samples were dropped from the majority class to balance the dataset."
        )

        # Separate the balanced data and labels
        balanced_X = balanced_dataset.drop(columns=["label"])
        balanced_y = balanced_dataset["label"]

        return np.array(balanced_X), np.array(balanced_y)

    def _closest_smaller_divisible_by_3(self, number: int) -> int:
        # Find the remainder when dividing the number by 3
        remainder = number % 3

        # Calculate the difference needed to reach the previous multiple of 3
        difference = remainder

        # Calculate the closest smaller number that is divisible by 3
        closest_smaller = number - difference

        return closest_smaller

    def _get_image_size(self, data: np.ndarray) -> int:
        return self._closest_smaller_divisible_by_3(data.shape[1])

    def _create_model(self, image_size: int) -> Model:
        # Define the input shape (for ResNet-50)
        input_shape = (image_size, image_size, 3)

        # Create a pre-trained ResNet-50 models
        # base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)

        # Freeze the layers in the base models
        for layer in base_model.layers:
            layer.trainable = True

        # Add custom layers for your specific task
        x = base_model.output
        x = Flatten()(x)
        x = Dense(image_size, activation="relu")(x)
        x = Dense(image_size, activation="relu")(x)
        output = Dense(self.num_classes, activation="softmax")(x)

        # Create the final models
        model = Model(inputs=base_model.input, outputs=output)

        # loss = 'categorical_crossentropy'

        # Compile the models
        model.compile(
            optimizer=SGD(learning_rate=0.001, momentum=0.9),
            loss=self.loss,
            metrics=["accuracy"],
        )

        # Optionally, you can print a summary of the models architecture
        model.summary()
        return model

    def _make_predictions_binary(self, predictions: np.ndarray) -> np.ndarray:
        res = []
        for p in predictions:
            if p[0] > p[1]:
                res.append(0)
            else:
                res.append(1)
        return np.array(res)


class GAFDataGenerator(Sequence):
    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        labels: Union[np.ndarray, pd.DataFrame],
        batch_size: int,
        image_size: int,
    ):
        self.data = np.array(data)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.image_size = image_size
        self.gaf = GramianAngularField(image_size=image_size)

    def __len__(self) -> int:
        ln = int(np.ceil(len(self.data) / self.batch_size))
        return ln

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_data = self.data[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx].reshape(-1, 1)

        transformed_data = []
        for row in batch_data:
            row = row.copy()
            row_data = np.array(row).reshape(1, -1)
            gaf_image = self.gaf.fit_transform(row_data)
            gaf_image = gaf_image.reshape(self.image_size, self.image_size, 1)
            gaf_image = np.repeat(gaf_image, 3, axis=-1)
            transformed_data.append(gaf_image)

        return np.array(transformed_data), np.array(batch_labels)
