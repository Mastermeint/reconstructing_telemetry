import numpy as np
from scipy import stats, signal, fft
from statsmodels.regression.linear_model import yule_walker
from functions import produce_differences_df, spec_pgram, spec_taper, spec_ci, spec_ar, plot_spec, get_quads, compute_df_frequencies
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.tsaplots as tsaplots
from statsmodels.tsa.arima_model import ARMA
from statsmodels.compat import lzip
from scipy import stats
import seaborn as sns

from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn as skl
#
# Load IRIS dataset
#
#iris = datasets.load_iris()
#X = iris.data
df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/spectral_multiple_frequencies.csv')
first_column = df.columns[0]

# Delete first
X = df.drop([first_column], axis=1)

X = X.to_numpy()
print(X)

#y = iris.target
#
# Instantiate the KMeans models
#
km = KMeans(n_clusters=20, random_state=42)
#
# Fit the KMeans model
#
km.fit_predict(X)
centroids = km.cluster_centers_
print('Here you hae the centroids of the clusters: ')
print(centroids)

#
# Calculate Silhoutte Score
#
score = skl.metrics.silhouette_score(X, km.labels_, metric='euclidean')
#
# Print the score
#
print('Silhouetter Score: %.3f' % score)


from yellowbrick.cluster import SilhouetteVisualizer

fig, ax = plt.subplots(2, 2, figsize=(15,8))

for i in [2, 3, 4, 5]:
    '''
    Create KMeans instance for different number of clusters
    '''
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    q, mod = divmod(i, 2)
    '''
    Create SilhouetteVisualizer instance with KMeans instance
    Fit the visualizer
    '''
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(X)


plt.show()


km = KMeans(n_clusters=20, init='k-means++', n_init=10, max_iter=100, random_state=42)

visualizer = SilhouetteVisualizer(km, colors='yellowbrick')
visualizer.fit(X)

plt.title('Silhoutte Visualizers for Spectral Analysis 10 Clusters all customers')
plt.show()


kmeans = KMeans(n_clusters=20, init = 'k-means++', n_init = 10, max_iter = 100, random_state=42).fit(X)

print(kmeans.labels_)
labels = kmeans.labels_
print(kmeans.cluster_centers_)

df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/spectral_multiple_frequencies.csv')

df['K-means cluster'] = labels

print(df.head())

df.to_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/spectral_multiple_frequencies_cluster.csv')