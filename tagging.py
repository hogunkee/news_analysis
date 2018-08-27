
class tagging:
    def __init__(self, tfidf, dic, phraser):
        self.tfidf = tfidf
        self.dic = dic
        self.phraser = phraser
    custom_tfidf = Model(dic=dic, name='0.3', wlocal=wlocal, wglobal=wglobal).get_model
    aaa = Extractor(custom_tfidf, dic, phraser).extract("PEP", topn1=10, topn2=200)
    c_dict = {}
    t = sum([j for i,j in aaa])**0.5
    for ticker, num in aaa:
        tmp = cluster.get_cluster(ticker)
        if tmp[0] not in c_dict.keys():
            c_dict[tmp[0]] = np.exp(num/t)
            continue
        c_dict[tmp[0]] += np.exp(num/t)

    high = sorted([(i,j) for i, j in c_dict.items()], key=lambda x: x[1], reverse=True)
    
    for i, j in high[:10]:
        print(cluster.get_keyword(i))
        print("-"*150)
