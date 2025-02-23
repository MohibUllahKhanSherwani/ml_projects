# -*- coding: utf-8 -*-
"""Customer Segregation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PfScOlstsTWcASsseM8bOw2UcSZBozsS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import preprocessing

data = pd.read_csv('/content/Mall_Customers.csv')

data.head()

data.shape

data.describe()

data.isnull().sum()

data.info()

encoder = preprocessing.LabelEncoder()

data['Gender'].value_counts()

data['Gender'] = encoder.fit_transform(data['Gender'])

data['Gender'].value_counts()

data.head()

#We are taking 2 columns -> Annual Income, Spending Score
x = data.iloc[:,[3, 4]].values

print(x)

#Choose k = number of clusters
# WCSS -> Within Clusters Sum of Squares
#Finding WCSS value for diff number of clusters
wcss = []
for i in range(1, 11):
  kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
  kmeans.fit(x)
  wcss.append(kmeans.inertia_)

# plot an elbow graph
sns.set()
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

"""We can see there are sharp drops at Number of clusters = 4 and 5 and also at 6, so 6 is optimal"""

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 2)

#Return a label for each data point based on their cluster 0-5
y = kmeans.fit_predict(x)

print(y)

#Visualizing the clusters
plt.figure(figsize = (10, 10))
plt.scatter(x[y == 0, 0], x[y == 0, 1], s = 50, c = 'green', label = 'Cluster 1')
plt.scatter(x[y == 1, 0], x[y == 1, 1], s = 50, c = 'red', label = 'Cluster 2')
plt.scatter(x[y == 2, 0], x[y == 2, 1], s = 50, c = 'blue', label = 'Cluster 3')
plt.scatter(x[y == 3, 0], x[y == 3, 1], s = 50, c = 'black', label = 'Cluster 4')
plt.scatter(x[y == 4, 0], x[y == 4, 1], s = 50, c = 'pink', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'indigo', label = 'Centroids')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()

