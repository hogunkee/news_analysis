import pickle
from ThemeSeacher import PhraserModel, tokenizer, clean, Clustering


phraser = PhraserModel().get_phraser
cluster = Clustering()
theme_dict = pickle.load(open('tag_cluster.bin', 'rb'))
def search(keyword):
    word = [i for i in phraser[tokenizer([clean(keyword)])]][0][0]
    cluster_num = cluster.get_cluster(word)[0]
    result =  sorted(theme_dict[cluster_num], key=lambda x: x[1], reverse=True)
        
    for i in result:
        info = "ticker : {0} \nsocre: {1:.2f} \nfull_name : {2} \nsector : {3} \nindustrty : {4} \n".format(i[0],i[1],i[2],i[3],i[4])
        print(info)
        
    if result == []:
        print("Nothing Matched!")