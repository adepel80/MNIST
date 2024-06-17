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
![IRIS DATASET](https://github.com/adepel80/MNIST/assets/123180341/0fb61ead-763c-4f8f-9a2d-1b85aad488e9)
```
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version = 1, return_X_y=True)

```
![mnist import dataset](https://github.com/adepel80/MNIST/assets/123180341/4729217b-afe9-4bc1-b914-372294320ac1)

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
![mnist reshape](https://github.com/adepel80/MNIST/assets/123180341/3db7a6f1-226d-4765-a28a-36746db1a66e)

# BAGGING ALGORITHM
```
#bagging algorithm
from sklearn.ensemble import BaggingClassifier
BaggingClassifier
```
![importing bagging classifier](https://github.com/adepel80/MNIST/assets/123180341/a71f10d6-dbfb-4039-8209-2ac8824c9e5f)

```
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
bag = BaggingClassifier(knn, max_samples =.5, max_features=28, n_estimators=20)
```
![mnist knnbag](https://github.com/adepel80/MNIST/assets/123180341/a074acbe-b97d-4b56-a62d-7df01950c8cf)

# TRAIN THE BAG ALGORITHM
```
bag.fit(X_train, y_train)
```
# ACCURACY FOR BAGGING PREDICTIONS
```
y_pred = bag.predict(X_test)
accuracy_score(y_test, y_pred)
```
![mnist bag acc score](https://github.com/adepel80/MNIST/assets/123180341/796904c0-b42e-4d02-bb10-08d0597a8eaf)
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
![mnist Matplot visual](https://github.com/adepel80/MNIST/assets/123180341/f5dcb7fb-631a-4d3d-a436-beeff1e5e9d1).


