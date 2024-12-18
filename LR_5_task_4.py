import argparse 
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn import datasets
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
def visualize_classifier(classifier, X, y, title=''):
    # Define the minimum and maximum values for X and Y
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # Define the step size for plotting
    mesh_step_size = 0.01

    # Define the mesh grid of X and Y values
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size),
                                 np.arange(min_y, max_y, mesh_step_size))

    # Run the classifier on the mesh grid
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)

    # Create the plot
    plt.figure()
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.Paired, shading='auto')

    # Overlay the training points on the plot
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1,
                cmap=plt.cm.Paired)

    # Set the title and axes
    plt.title(title)
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())

    # Show the plot
    plt.show()
# Завантаження вхідних даних
housing_data = fetch_california_housing()

X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Train AdaBoost regressor
regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
regressor.fit(X_train, y_train)

# Predict and evaluate performance
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
print("\nADABOOST REGRESSOR")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

# Feature importance visualization
feature_importances = regressor.feature_importances_
feature_names = housing_data.feature_names

# Normalize the feature importances
feature_importances = 100 * (feature_importances / max(feature_importances))

# Sort features by importance
index_sorted = np.flipud(np.argsort(feature_importances))

# Generate positions for the bar chart
pos = np.arange(index_sorted.shape[0]) + 0.5

# Plotting
plt.figure()
plt.bar(pos, feature_importances[index_sorted], align='center')
plt.xticks(pos, np.array(feature_names)[index_sorted])  # Convert feature_names to a NumPy array
plt.ylabel('Relative Importance')
plt.title('Feature Importance with AdaBoost Regressor')
plt.show()