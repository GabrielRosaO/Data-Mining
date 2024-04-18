from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay
from sklearn import metrics
import seaborn as sns
from itertools import cycle

#Import dataset Iris and show the head of the dataset with help of a dataframe in Pandas library
iris = datasets.load_iris()
dataFrame = pd.DataFrame(data = iris.data, columns = iris.feature_names)
dataFrame.head()


#define variables, add some "noisy to the dataset with random samples and split into training set and test set"
X, y = iris.data, iris.target
class_names = iris.target_names

random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
n_classes = len(np.unique(y))
X = np.concatenate([X, random_state.randn(n_samples, 50 * n_features)], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)


#declare the model and train it with the data
knn = KNeighborsClassifier(n_neighbors= 5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_score = knn.predict_proba(X_test)

#find the confusion matrix of the model and show the main results,(Precision, Recall, F1-Score and total accurancy)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)#puts the matrix in the variable to display in the next section

print(metrics.classification_report(y_test, y_pred))

#Show the confusion matrix in a heat map
plt.figure(figsize=(3,3))
sns.heatmap(confusion_matrix, annot = True, fmt = '0.1f', linewidth = 0.5, square = True, cbar = False)
plt.ylabel("Actual values")
plt.xlabel("Predicted values")

plt.show()


#Split the data to identify and select one class of interest to build the ROC curve
label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_test)
y_onehot_test.shape 

#Find the ID of the class that will be used in the ROC curve, the utilized method is the One Vs Rest
class_of_interest = "virginica"

if(class_of_interest == "setosa"):
  class_ = 0
elif(class_of_interest == "versicolor"):
  class_ = 1
elif(class_of_interest == "virginica"):
  class_ = 2

class_id = np.flatnonzero(label_binarizer.classes_ == class_)[0]
class_id#id of the class in the splitted dataset

#Show the ROC Curve in One Vs Rest
display = RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    y_score[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
    plot_chance_level=True,
)
_ = display.ax_.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="One-vs-Rest ROC curves:\nVirginica vs (Setosa & Versicolor)",
)
plt.show()

#Show all the ROC Curves of each class in one graph

fig, ax = plt.subplots(figsize=(5,5))

colors = cycle(["aqua", "darkorange", "cornflowerblue"])

for class_id, color in zip(range(n_classes),colors):
  RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    y_score[:, class_id],
    name=f"ROC curve for {class_names[class_id]}",
    color=color,
    ax=ax,
    plot_chance_level= (class_id == 2),
)
_ = ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="One-Vs_Rest ROC curves for each class",
)
plt.show()
