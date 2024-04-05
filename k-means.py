from sklearn.cluster import KMeans
import numpy as np

data = np.array([[20, 500], [40, 1000], [30, 800], [18, 300], [28, 1200], [35, 1400], [45, 1800]])
k=2

kmeans = KMeans(n_clusters = k)
kmeans.fit(data)
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_
print("Cluster Centers: ")
for i, center in enumerate(cluster_centers):
    print("Cluster ", i+1, " Center: ", center)
print("\nLabels: ")
for i,label in enumerate(labels):
    print("Data Point ", i+1, " is in Cluster ", label+1)

#     Cluster Centers: 
# Cluster  1  Center:  [  37. 1350.]
# Cluster  2  Center:  [ 22.66666667 533.33333333]

# Labels:
# Data Point  1  is in Cluster  2
# Data Point  2  is in Cluster  1
# Data Point  3  is in Cluster  2
# Data Point  4  is in Cluster  2
# Data Point  5  is in Cluster  1
# Data Point  6  is in Cluster  1
# Data Point  7  is in Cluster  1