import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


dir_news = "news/"
dir_cleaned_news = "cleaned_news/"
dir_phraser = "phraser/"
dir_dictionary = "dictionary/"
dir_tfidf = "tfidf/"
dir_embedding = "embedding/"
dir_clustering = "clustering/"

embedding = EmbeddingModel().get_embedding

class Clustering:
    def __init__(self, name="kmeans", n_clusters=2000, embedding=embedding):
        self.name = name + "bin"
        self.n_clusters = n_clusters
        if self.name in os.listdir(dir_clustering):
            with open(dir_clustering + self.name, "rb") as c:
                print(self.name, " loaded")
                self.get_cluster = pickle.load(c)
        else:
            print("cluster not exists")
            print("start clustering...")
            self.clustering()
            self.save()
    
    def loading_embedding(self):
        for idx, wid in enumerate(dic):
            if idx % 10000 == 0:
                print(idx)
            if idx == 0:
                first = embedding.wv[dic[wid]]
                continue
        try:
            first = np.vstack((first, embedding.wv[dic[wid]]))
        except:
            continue
        print(first.shape)
        np.save(dir_clustering+'emb_dict.npy', first)


    def clustering(self):
        np.random.seed(42)
        if 'emb_dict.npy' not in os.listdir(dir_clustering):
            loading_embedding()
            
        data = np.load(dir_clustering+'emb_dict.npy')

        n_words, n_features = data.shape

        print("n_clusters: %d, \t n_words %d, \t n_features %d"
          % (self.n_clusters, n_words, n_features))
        self.get_cluster = KMeans(init='k-means++', n_clusters=self.n_clusters, n_init=10)
        self.get_cluster.fit(data)
    
    def save(self):
        pickle.dump(self.get_cluster, open(dir_clustering+self.name, 'wb'))