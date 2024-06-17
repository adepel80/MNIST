# MNIST
Implementing the MNIST digit recognition using a bagging algorithm to improve classification accuracy through ensemble learning.
# LIBRARIES
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chart_studio.plotly as py
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,r2_score
plt.style.use ("dark_background")

import os
```
# LOAD THE MNIST DATASET
```
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version = 1, return_X_y=True)

```
# TRAIN AND TEST THE DATASET
```

from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X,y,test_size =1/7, random_state=0)

```
# RESHAPE THE SCALE TO BE IN [0, 1]
```
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
#reshape and scale to be in [0,1]
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
```

# BAGGING ALGORITHM
```
#bagging algorithm
from sklearn.ensemble import BaggingClassifier
BaggingClassifier
```
```
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
bag = BaggingClassifier(knn, max_samples =.5, max_features=28, n_estimators=20)
```
# TRAIN THE BAG ALGORITHM
```
bag.fit(X_train, y_train)
```
# ACCURACY FOR BAGGING PREDICTIONS
```
y_pred = bag.predict(X_test)
accuracy_score(y_test, y_pred)
```
# IMPORT CLASSIFICATION REPORT
```
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, labels=bag.classes_.tolist()))
```
# CONFUSION MATRIX FOR IRIS DATASET
```
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from sklearn.metrics import classification_report
#assuming 'knn' is your trained model, 'X_test' are your test features
predictions = bag.predict(X_test)
cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=bag.classes_)
disp.plot()

plt.suptitle("Confusion Matrix for Iris Dataset")
plt.show()
```
