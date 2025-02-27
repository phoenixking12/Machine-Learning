import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import kagglehub

data = pd.read_csv(kagglehub.dataset_download("vjchoudhary7/customer-segmentation-tutorial-in-python") + "/Mall_Customers.csv")
data = data.drop(columns=['CustomerID'])

#data = pd.read_csv("Mall_Customers.csv")

if 'Gender' in data.columns:
    data['Gender'] = LabelEncoder().fit_transform(data['Gender'])

scaler = StandardScaler()
scaledData = scaler.fit_transform(data)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(scaledData)

pca = PCA(2)
df_pca = pca.fit_transform(scaledData)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=data['Cluster'], palette='Set1', s=100)
plt.title('K-means Clustering of Customers')
plt.show()
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=data.columns[:-1])
print("Cluster Centers:\n", cluster_centers)
