import argparse 
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report

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
input_file= 'data_imbalance.txt'
data = np.loadtxt(input_file,delimiter=',')
X,y = data[:,:-1],data[:,-1]
# Поділ вхідних даних на два класи на підставі міток
class_0=np.array(X[y==0])
class_1=np.array(X[y==1])
# Візуалізація вхідних даних
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='black', edgecolors='black', linewidth=1, marker='x')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white', edgecolors='black', linewidth=1, marker='o')
plt.title('Входные данные')

# Розбиття даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.25, random_state=5)

# Класифікатор на основі гранично випадкових лісів
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
if len(sys.argv) > 1:
    if sys.argv[1] == 'balance':
        params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0, 'class_weight': 'balanced'}
else:
    raise TypeError ("Invalid input argument; should be 'balance'")


classifier=ExtraTreesClassifier (**params)
classifier.fit(X_train, y_train)
visualize_classifier(classifier, X_train, y_train, 'Training dataset')


y_test_pred=classifier.predict (X_test)
visualize_classifier(classifier, X_test, y_test, 'Teстовый набор данныx')

# Обчислення показників ефективності класифікатора
class_names = ['Class-0', 'Class-1'] 
print("\n" + "#"*40)
print("\nClassifier performance on training dataset\n") 
print (classification_report (y_train,
classifier.predict (X_train), target_names=class_names))
print("#"*40 + "\n")
print("#"*40)
print("\nClassifier performance on test dataset\n") 
print(classification_report (y_test, y_test_pred, target_names=class_names))
print("#"*40 + "\n")
plt.show()