import os
import time
import pickle
import pandas as pd
import numpy as np
from gensim.parsing.preprocessing import *
from gensim.utils import lemmatize
from gensim.models import TfidfModel, FastText
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases, Phraser
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


class Extractor:
    def __init__(self, model, dic, phraser=None):
        self.model = model
        self.dic = dic        
        self.phraser = phraser
        
    def corpus(self, ticker):
        df = pd.read_csv(dir_cleaned_news+ticker, index_col=0)
        corpus = []
        for tokenized_doc in tokenizer(df['content'], self.phraser): 
            corpus += [self.dic.doc2bow(tokenized_doc)]
        return corpus

    def extract(self, ticker, topn1=10, topn2=10):
        check = {}

        for doc in self.corpus(ticker+".csv"):
            vector = self.model[doc]
            top = sorted(vector, key=lambda x:x[1], reverse=True)[:topn1]
            for word in [self.dic[i[0]] for i in top]:
                if word not in check.keys():
                    check[word] = 0
                check[word] += 1
            
        checklist = []
        for i in check.items():
            checklist.append(i)
        
        return sorted(checklist, key=lambda x: x[1], reverse=True)[:topn2]
    

cluster = Clustering()
    
class Extractor_cluster(Extractor):
    def extract(self, ticker, cluster, topn1=10, topn2=10):
        check = {}

        for doc in self.corpus(ticker+".csv"):
            vector = self.model[doc]
            top = sorted(vector, key=lambda x:x[1], reverse=True)[:topn1]
            for word in [self.dic[i[0]] for i in top]:
                c = cluster.get_cluster(word)
                if c[0] not in check.keys():
                    check[c[0]] = 0
                check[c[0]] += 1
            
        checklist = []
        for i in check.items():
            checklist.append(i)
        
        return sorted(checklist, key=lambda x: x[1], reverse=True)[:topn2]