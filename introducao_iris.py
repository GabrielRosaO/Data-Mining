from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
k = 5  # Número de vizinhos
knn = KNeighborsClassifier(n_neighbors=k)

#Vetores para guardar rótulos verdadeiros e probabilidades previstas
true_labels = []
predicted_probs = []

for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    knn.fit(X_train, y_train)  # Treinando o modelo
    y_pred = knn.predict(X_test)  # Prevendo rotulos de dados de teste
    y_prob = knn.predict_proba(X_test)  # Prevendo probabilidade de rotulos de dados de teste

    #Calculando métricas de avaliação
    accuracy = accuracy_score(y_test, y_pred)
    precision_versicolor = precision_score(y_test, y_pred, labels=[1], average=None)[0]
    recall_versicolor = recall_score(y_test, y_pred, labels=[1], average=None)[0]
    f1_versicolor = f1_score(y_test, y_pred, labels=[1], average=None)[0]

    #Adicionando rótulos verdadeiros e probabilidades previstas
    true_labels.extend(y_test)
    predicted_probs.extend(y_prob)

    #plot das métricas
    fig, (ax1, ax2) = plt.subplots(1, 2)

    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height

    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True, title= f"Matriz de confusão KNN {i}", ax=ax1)

    ax2.text(0.1, 0.9, f"Acurácia: {accuracy:.2f}", fontsize=12, ha='left', transform=ax2.transAxes)
    ax2.text(0.1, 0.8, f"Precisão Versicolor: {precision_versicolor:.2f}", fontsize=12, ha='left', transform=ax2.transAxes)
    ax2.text(0.1, 0.7, f"Recall Versicolor: {recall_versicolor:.2f}", fontsize=12, ha='left', transform=ax2.transAxes)
    ax2.text(0.1, 0.6, f"F1-score Versicolor: {f1_versicolor:.2f}", fontsize=12, ha='left', transform=ax2.transAxes)

    ax2.axis('off')

skplt.metrics.plot_roc(true_labels, predicted_probs)

plt.show()


