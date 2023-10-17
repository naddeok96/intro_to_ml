import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder

# Create folder if it doesn't exist
if not os.path.exists('sklearn_figures'):
    os.makedirs('sklearn_figures')

# Load Iris Data
df = pd.read_csv("iris.csv")

# Prepare Data
X = df.drop('class', axis=1)
y = df['class']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, random_state=42)

# Logistic Regression: Confusion Matrix
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_cm = confusion_matrix(y_test, lr.predict(X_test))
ConfusionMatrixDisplay(lr_cm, display_labels=label_encoder.classes_).plot()
plt.title('Confusion Matrix for Logistic Regression')
plt.savefig('sklearn_figures/logistic_regression_cm.png')
plt.close()

# Decision Tree: Decision Boundary
dt = DecisionTreeClassifier()
dt.fit(X_train[['petal_length', 'petal_width']], y_train)
# dt.fit(X_train[['sepal_length', 'sepal_width']], y_train)
xx, yy = np.meshgrid(np.linspace(X['sepal_length'].min(), X['sepal_length'].max(), 100),
                     np.linspace(X['sepal_width'].min(), X['sepal_width'].max(), 100))

# xx, yy = np.meshgrid(np.linspace(X['petal_length'].min(), X['petal_length'].max(), 100),
#                      np.linspace(X['petal_width'].min(), X['petal_width'].max(), 100))
Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8)
for label in np.unique(y_encoded):
    plt.scatter(X_train['sepal_length'][y_train == label], 
                X_train['sepal_width'][y_train == label], 
                label=label_encoder.classes_[label])

plt.title('Decision Boundary for Decision Tree')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.savefig('sklearn_figures/decision_tree_boundary.png')
plt.close()

# Random Forest: Feature Importance
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure()
plt.title('Feature Importances for Random Forest')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), X.columns[indices], rotation=45)
plt.savefig('sklearn_figures/random_forest_feature_importance.png')
plt.close()

# Support Vector Machine: Precision-Recall Curve
# Support Vector Machine: Confusion Matrix
svc = SVC()
svc.fit(X_train, y_train)
svc_cm = confusion_matrix(y_test, svc.predict(X_test))
ConfusionMatrixDisplay(svc_cm, display_labels=label_encoder.classes_).plot()
plt.title('Confusion Matrix for Support Vector Machine')
plt.savefig('sklearn_figures/support_vector_machine_cm.png')
plt.close()

