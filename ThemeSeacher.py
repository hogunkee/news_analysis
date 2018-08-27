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



def tokenizer(corpus, phraser=None):
    tokenized = []
    for text in corpus:
        if type(text) is str:
            if phraser:
                tokenized += [token for token in phraser[[text.split()]]]
            else:
                tokenized += [text.split()]
        else:
            continue
    return tokenized


# 텍스트 전처리 함수
def clean(text):
    text = strip_multiple_whitespaces(strip_non_alphanum(text)).split()
    words = []
    for word in text:
        tmp = lemmatize(word)
        if tmp:
            words.append(tmp[0][:-3].decode("utf-8"))
    return " ".join(words)


class Updator():
    def update(self):
        print("Updating...")
        for ticker in os.listdir(dir_news):
            if ticker in os.listdir(dir_cleaned_news):
                continue
            if ticker.endswith(".csv"):
                start = time.time()
                print("cleaning ", ticker)
                df = pd.read_csv(dir_news+ticker, index_col=0)
                df["content"] = df["content"].apply(clean)
                df.to_csv(dir_cleaned_news+ticker)
                end = time.time()
                print(ticker, " done! ", end-start, " seconds")



                
class PhraserModel():
    def __init__(self, name="default"):
        self.name = "phraser_" + name + ".bin"
        if self.name in os.listdir(dir_phraser):
            with open(dir_phraser + self.name, "rb") as p:
                print(self.name, " loaded")
                self.get_phraser = pickle.load(p)
        else:
            print("phraser not exists")
            print("start building...")
            self.build_phraser()
            self.save()
            
    def build_phraser(self):
        tickers = [i for i in os.listdir(dir_cleaned_news) if i.endswith(".csv")]
        corpus = []
        start = time.time()
        for ticker in tickers:
            df = pd.read_csv(dir_cleaned_news + ticker, index_col=0)
            corpus += tokenizer(df['content'])
        
        
        self.get_phraser = Phraser(Phrases(corpus))
        end = time.time()
        print("train finished! ", end-start, " seconds")
    
    # 저장
    def save(self):
        with open(dir_phraser + self.name, "wb") as p:
            pickle.dump(self.get_phraser, p)
        print("saved!")

        
class KeywordDict():
    # dictionary.bin 있으면 로드하고 없으면 새로 만들고 저장
    def __init__(self, name="default", phraser=None):
        self.name = "dictionary_" + name + ".bin"
        self.phraser = phraser
        if self.name in os.listdir(dir_dictionary):
            with open(dir_dictionary+self.name, 'rb') as dic:
                self.get_dict = pickle.load(dic)
            print("keyword dictionary loaded")
        else:
            print("dictionary not exists")
            print("start building...")
            self.build_dictionary()
            self.save()

    # 사전 만드는 함수
    def build_dictionary(self):
        self.get_dict = Dictionary()
        tickers = [i for i in os.listdir(dir_cleaned_news) if i.endswith(".csv")]
        
        for ticker in tickers:
            df = pd.read_csv(dir_cleaned_news + ticker, index_col=0)
            self.get_dict.add_documents(tokenizer(df['content'], self.phraser))  
            print(ticker + " added")
        print("done")

    # 저장
    def save(self):
        with open(dir_dictionary+self.name, "wb") as dic:
            pickle.dump(self.get_dict, dic)
            
            
class Model:
    def __init__(self, dic, name="default", wlocal=None, wglobal=None, smartirs=None, phraser=None):
        self.name = "model_"+name+".bin"
        self.wlocal = wlocal
        self.wglobal = wglobal
        self.smartirs = smartirs
        self.dic = dic
        self.phraser = phraser
        name_list = []
        for i in 'nlabv':
            for j in 'ntp':
                for k in 'nc':
                    name_list.append(i+j+k)
        if self.name in os.listdir(dir_tfidf):
            with open(dir_tfidf+self.name, 'rb') as m:
                self.get_model = pickle.load(m)
            
            print("Model {} loaded".format(name))
        else:
            print("model not exists")
            print("start building...")
            self.build_model()
            self.save()
            
    def build_model(self):
        start = time.time()
        tickers = [i for i in os.listdir(dir_cleaned_news) if i.endswith(".csv")]
        corpus = []
        for ticker in tickers:
            df = pd.read_csv(dir_cleaned_news + ticker, index_col=0)
            for tokenized_doc in tokenizer(df['content'], self.phraser): 
                corpus += [self.dic.doc2bow(tokenized_doc)]
        
        if self.wlocal and self.wglobal:
            self.get_model = TfidfModel(corpus, dictionary=self.dic, 
                                    wlocal=self.wlocal , wglobal=self.wglobal, smartirs=self.smartirs)
        else:
            self.get_model = TfidfModel(corpus, dictionary=self.dic, smartirs=self.smartirs)
            
        end = time.time()

        print(self.name, " finished ", end-start, " seconds")
    
    def save(self):
        with open(dir_tfidf + self.name, "wb") as m:
            pickle.dump(self.get_model, m)
            
            
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
    

class EmbeddingModel():
    def __init__(self, name="default", phraser=None):
        self.name = "embedding_" + name + ".model"
        self.phraser = phraser
        if self.name in os.listdir(dir_embedding):
            self.get_embedding = FastText.load(dir_embedding+self.name)
            print("Embedding {} loaded".format(name))
        else:
            print("embedding not exists")
            print("start building...")
            self.build_embedding()
            self.save()
            
    def build_embedding(self):
        tickers = [i for i in os.listdir(dir_cleaned_news) if i.endswith(".csv")]
        tokenized_docs = []
        start = time.time()
        for ticker in tickers:
            df = pd.read_csv(dir_cleaned_news + ticker, index_col=0)
            tokenized_docs += tokenizer(df['content'], self.phraser)
        
        self.get_embedding = FastText(tokenized_docs, sg=1, hs=1)
        end = time.time()
        
        print("train finished! ", end-start, " seconds")
    
    # 저장
    def save(self):
        self.get_embedding.save(dir_embedding+self.name)
        print("saved!")

        

embedding = EmbeddingModel().get_embedding

class Clustering:
    def __init__(self, name="kmeans", n_clusters=2000, embedding=embedding):
        self.embedding = embedding
        self.name = name + ".bin"
        self.n_clusters = n_clusters
        if self.name in os.listdir(dir_clustering):
            with open(dir_clustering + self.name, "rb") as c:
                print(self.name, " loaded")
                self.get_model = pickle.load(c)
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
        self.get_model = KMeans(init='k-means++', n_clusters=self.n_clusters, n_init=10)
        self.get_model.fit(data)
    
    def get_cluster(self, keyword):
        return self.get_model.predict([self.embedding.wv[keyword]])
    
    def get_keyword(self, cluster, topn=10):
        return self.embedding.wv.similar_by_vector(self.get_model.cluster_centers_[cluster], topn = topn)
    
    def get_keyword_by_word(self, word, topn=10):
        return self.get_keyword(self.get_cluster(word)[0], topn)
    
    def save(self):
        pickle.dump(self.get_model, open(dir_clustering+self.name, 'wb'))
        
class Tagging:
    def __init__(self):
        self.phraser = PhraserModel().get_phraser
        self.dic = KeywordDict().get_dict
        self.tfidf = Model(self.dic, name='vpn').get_model
        self.extract = Extractor(self.tfidf, self.dic, self.phraser)
        self.cluster = Clustering()
        # self.keyword = keyword
        
    def ticker2cluster(self, ticker, topn=15):
        keywords = self.extract.extract(ticker, topn1=10, topn2=100)
        c_dict = {}
        
        # for normalize
        t = sum([j for i,j in keywords])**0.5
        
        for ticker, num in keywords:
            tmp = self.cluster.get_cluster(ticker)
            if tmp[0] not in c_dict.keys():
                c_dict[tmp[0]] = np.exp(num/t)
                continue
            c_dict[tmp[0]] += np.exp(num/t)
            
        return sorted([(i,j) for i, j in c_dict.items()], key=lambda x: x[1], reverse=True)[:topn]
    
    def cluster2keyword(self, t2c_output):
        for cluster, score in t2c_output:
            print(self.cluster.get_keyword(cluster))
            print("-"*150)
            
    def tagging(self, symbols):
        df = pd.read_csv('ticker.csv')
        tag = {}
        for idx, symbol in enumerate(symbols):
            print('tagging... ', symbol, end=' ')
            tag[symbol] = {'cluster': self.ticker2cluster(symbol), 
                           'name': df['name'][idx] , 
                           'sector': df['sector'][idx] , 
                           'industry': df['industry'][idx]}
            print('done! {}/505...'.format(idx+1))
        return tag
                       
            
    def tagging_cluster(self, tagged_dict):
        tag = {}
        for i in range(2000):
            tag[i] = []
        
        idx = 0
        for ticker, subdict in tagged_dict.items():
            print('tagging... ', ticker, end=' ')
            for cluster, score in subdict['cluster']:
                name = subdict['name']
                sector = subdict['sector']
                industry = subdict['industry']
                tag[cluster].append((ticker, score, name, sector, industry))
            idx += 1
            print('done! {}/505...'.format(idx))
            
        return tag

    
        
            
if __name__=='__main__':
    print('main')
    phraser = PhraserModel().get_phraser
    dic = KeywordDict(phraser=phraser).get_dict
    tfidf = Model(dic, name='vpc', smartirs='Lpc', phraser=phraser).get_model
    Extractor(tfidf, dic, phraser).extract("NVDA", topn1=10, topn2=10)
    print('end')
            
            
            
            
