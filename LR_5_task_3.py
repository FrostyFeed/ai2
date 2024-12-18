import argparse 
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
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
input_file= 'data_random_forests.txt'
data = np.loadtxt(input_file,delimiter=',')
X,y = data[:,:-1],data[:,-1]
# Поділ вхідних даних на 3 класи на підставі міток
class_0=np.array(X[y==0])
class_1=np.array(X[y==1])
class_2=np.array(X[y==2])
# Розбиття даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.25, random_state=5)

# Визначення сітки значень параметрів
parameter_grid=[{'n_estimators': [100],'max_depth': [2, 4, 7, 12, 16]},{ 'max_depth': [4], 'n_estimators': [25, 50, 100, 250]}]
metrics=['precision_weighted','recall_weighted']


for metric in metrics:
    print("\n##### Searching optimal parameters for", metric)
    classifier=GridSearchCV(ExtraTreesClassifier(random_state=0),parameter_grid, cv=5, scoring=metric)
    classifier.fit(X_train, y_train)


print ("\nGrid scores for the parameter grid: ") 
for params, mean_score in zip(classifier.cv_results_['params'], classifier.cv_results_['mean_test_score']):
        print(params, '-->', round(mean_score, 3))
print ("\nBest parameters: ", classifier.best_params_)


y_pred=classifier.predict (X_test)
print ("\nPerformance report: \n")
print (classification_report (y_test, y_pred))

