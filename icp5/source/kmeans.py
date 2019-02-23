import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")


#reading data set
dataset = pd.read_csv('College.csv')
x = dataset.iloc[:,[5,6,7,8]]



#normalizing and preprocessing data
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.fit_transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)


X_scaled.sample(4)



nclusters = 4
seed = 0

km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(X_scaled)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(X_scaled)

# calculating silhouette score
from sklearn import metrics
score = metrics.silhouette_score(X_scaled, y_cluster_kmeans)
print('silhouette_score :', score)

#Plotting clusters.
LABEL_COLOR_MAP = {0 : 'y',
                   1 : 'r',
                   2 : 'b',
                   3 : 'g'
                   }
label_color = [LABEL_COLOR_MAP[l] for l in km.predict(X_scaled)]
plt.scatter(X_scaled_array[:, 0], X_scaled_array[:, 1], c=label_color)
plt.title("clustered based on 5, 6, 7, 8 columns")
plt.show()