from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import scikitplot as skplt
import matplotlib.pyplot as plt

#Carregando conjunto de dados
iris = load_iris()
X = iris.data  # Parâmetros
y = iris.target  # Rótulos

#Aplicando a validação cruzada
num_folds = 5  # Número de folds
kf = KFold(n_splits=num_folds, shuffle=True)

#Instanciando o classificador
k = 3  # Número de vizinhos
knn = KNeighborsClassifier(n_neighbors=k)

for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    knn.fit(X_train, y_train)  # Treinando o modelo
    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)

    #plot das métricas
    fig, ax = plt.subplots()

    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height

    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True, title= f"Matriz de confusão KNN {i}", ax=ax)
    skplt.metrics.plot_roc(y_test, y_prob, ax=ax)

    ax.text(0.5 * (left + right), bottom, "Texto de teste",
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes)

plt.show()


