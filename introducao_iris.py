from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import scikitplot as skplt
import matplotlib.pyplot as plt

#Carregando conjunto de dados
iris = load_iris()
X = iris.data  # Parâmetros
y = iris.target  # Rótulos

#Aplicando a validação cruzada
num_folds = 4  # Número de folds
kf = KFold(n_splits=num_folds, shuffle=True)

#Instanciando o classificadores
k = 5  # Número de vizinhos
knn = KNeighborsClassifier(n_neighbors=k)
dtc = DecisionTreeClassifier(random_state=42)

#Vetores para guardar rótulos verdadeiros e probabilidades previstas
true_labels = []
predicted_probs_knn = []
predicted_probs_dtc = []

for i, (train_index, test_index) in enumerate(kf.split(X)):

    # Separando dados em treino e teste
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #KNN treino e previsão
    knn.fit(X_train, y_train)  # Treinando o modelo
    y_pred_knn = knn.predict(X_test)  # Prevendo rotulos de dados de teste
    y_prob_knn = knn.predict_proba(X_test)  # Prevendo probabilidade de rotulos de dados de teste

    #DTC treino e previsão
    dtc.fit(X_train, y_train)  # Treinando o modelo
    y_pred_dtc = dtc.predict(X_test)  # Prevendo rotulos de dados de teste
    y_prob_dtc = dtc.predict_proba(X_test)  # Prevendo probabilidade de rotulos de dados de teste

    #Calculando métricas de avaliação KNN
    knn_accuracy = accuracy_score(y_test, y_pred_knn)
    knn_precision_versicolor = precision_score(y_test, y_pred_knn, labels=[1], average=None)[0]
    knn_recall_versicolor = recall_score(y_test, y_pred_knn, labels=[1], average=None)[0]
    knn_f1_versicolor = f1_score(y_test, y_pred_knn, labels=[1], average=None)[0]

    #Calculando métricas de avaliação DTC
    dtc_accuracy = accuracy_score(y_test, y_pred_dtc)
    dtc_precision_versicolor = precision_score(y_test, y_pred_dtc, labels=[1], average=None)[0]
    dtc_recall_versicolor = recall_score(y_test, y_pred_dtc, labels=[1], average=None)[0]
    dtc_f1_versicolor = f1_score(y_test, y_pred_dtc, labels=[1], average=None)[0]

    #Adicionando rótulos verdadeiros e probabilidades previstas
    true_labels.extend(y_test)
    predicted_probs_knn.extend(y_prob_knn)
    predicted_probs_dtc.extend(y_prob_dtc)

    #Plot
    fig, axes = plt.subplots(2, 3)

    #Matriz de confusão KNN
    skplt.metrics.plot_confusion_matrix(y_test, y_pred_knn, normalize=True, title= f"Matriz de confusão KNN {i}", ax=axes[0, 0])

    #Matriz de confusão DTC
    skplt.metrics.plot_confusion_matrix(y_test, y_pred_dtc, normalize=True, title= f"Matriz de confusão Arvore de decisão {i}", ax=axes[1, 0])

    #Métricas de avaliação KNN
    axes[0, 1].text(0.1, 0.9, f"Acurácia: {knn_accuracy:.2f}", fontsize=12, ha='left', transform=axes[0, 1].transAxes)
    axes[0, 1].text(0.1, 0.8, f"Precisão Versicolor: {knn_precision_versicolor:.2f}", fontsize=12, ha='left', transform=axes[0, 1].transAxes)
    axes[0, 1].text(0.1, 0.7, f"Recall Versicolor: {knn_recall_versicolor:.2f}", fontsize=12, ha='left', transform=axes[0, 1].transAxes)
    axes[0, 1].text(0.1, 0.6, f"F1-score Versicolor: {knn_f1_versicolor:.2f}", fontsize=12, ha='left', transform=axes[0, 1].transAxes)

    axes[0, 1].axis('off')
    axes[0, 1].set_title('Métricas de avaliação KNN')

    #Métricas de avaliação DTC
    axes[1, 1].text(0.1, 0.9, f"Acurácia: {dtc_accuracy:.2f}", fontsize=12, ha='left', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.8, f"Precisão Versicolor: {dtc_precision_versicolor:.2f}", fontsize=12, ha='left', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.7, f"Recall Versicolor: {dtc_recall_versicolor:.2f}", fontsize=12, ha='left', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.6, f"F1-score Versicolor: {dtc_f1_versicolor:.2f}", fontsize=12, ha='left', transform=axes[1, 1].transAxes)

    axes[1, 1].axis('off')
    axes[1, 1].set_title('Métricas de avaliação Arvore de decisão')

    #Visualização

    #KNN
    clf = Pipeline(
    steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=11))]
    )
    
    

    #DTC
    plot_tree(dtc, ax=[1, 2])

skplt.metrics.plot_roc(true_labels, predicted_probs_knn, title="Curva ROC KNN")
skplt.metrics.plot_roc(true_labels, predicted_probs_dtc, title="Curva ROC Arvore de decisão")

plt.show()


