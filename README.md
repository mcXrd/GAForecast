# GAForecast
This Python library is designed for making predictions on tabular data using a neural network based on ResNet50. The library's strength lies in its ability to excel in predicting time series data presented in a tabular format, where each row contains multiple historical time points.

## Source code
https://github.com/mcXrd/GAForecast

## Requirements
Prior to utilizing this library, please ensure that you have installed TensorFlow on your system.

Additionally, having access to a GPU is highly recommended.

## Memory requirements
Table with 70 features will require cca 7gb of GPU memory.

## Installation
You can install the GAForecast library from pypi:  https://pypi.org/project/GAForecast/


## Usage
```
from gaforecast.models.binary_classifier import GAForecastBinaryClassifier

# Create an instance of the GAForecastBinaryClassifier
clf = GAForecastBinaryClassifier()

# Fit the classifier to your tabular data
clf.fit(X_train, y_train)

# Make predictions on new data
y_pred = clf.predict(X_test)
```

## Internals
Transforms tabular data into images using the Gramian Angular Field technique.
Reshapes these images to fit into the ResNet50 architecture.
Builds a classifier on top of TensorFlow's ResNet50.
Offers a common scikit-learn interface for ease of use.


## Future Updates
In future updates, the library will include:

### 1) A regressor for continuous prediction tasks.
### 2) Support for multi-class classification.
Please stay tuned for these upcoming enhancements.

# License
This library is provided under the MIT License.

# Contact
For any questions or issues, please contact [matejka.vaclav@gmail.com].

# Acknowledgments
We would like to especially acknowledge the contributions of the pandas, numpy, scikit-learn, pytz, open-cv and TensorFlow communities for their invaluable libraries and resources.






