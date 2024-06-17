#!/usr/bin/env python
# coding: utf-8

# In[17]:


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


# In[18]:


from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version = 1, return_X_y=True)


# In[19]:


X.shape


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X,y,test_size =1/7, random_state=0)


# In[21]:


X_train.shape


# In[22]:


y_train.shape


# In[23]:


X_test.shape


# In[24]:


y_test.shape


# In[25]:


type(X_train)


# In[26]:


X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
#reshape and scale to be in [0,1]
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


# In[27]:


plt.figure(figsize=(20, 4))
for index in range(5):
    plt.subplot(1,5, index+1)
    plt.imshow(X_train[index].reshape((28,28)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % int(y_train.to_numpy()[index]), fontsize=20)


# In[28]:


#bagging algorithm
from sklearn.ensemble import BaggingClassifier
BaggingClassifier


# In[31]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
bag = BaggingClassifier(knn, max_samples =.5, max_features=28, n_estimators=20)


# In[32]:


bag.fit(X_train, y_train)


# In[33]:


y_pred = bag.predict(X_test)
accuracy_score(y_test, y_pred)


# In[34]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, labels=bag.classes_.tolist()))


# In[35]:


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


# In[ ]:




