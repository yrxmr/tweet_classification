from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize
import re


#obtention des vecteurs TF-IDF

def get_data(path_file, sep=','):
    return pd.read_csv(path_file, sep=sep).fillna(0)




def X_tfidf():
    stopwords = get_stop_words('french')
    data = get_data("tweetslabels.csv")
    doc= data['tweet']


#tokénisation avec NTLK
#néttoyage avec stopwords
#obtention d'une liste de listes
#chaque sous-liste est une liste de tokens

    sentences= []
    tokenized = []
    for element in doc:
        sentences.append(element)
    for sentence in sentences:
        sentence = word_tokenize(sentence)
        tokenized.append(sentence)


#on effectue la vectorisation sur l'ensemble des textes

    tf_idf_vec_smooth = TfidfVectorizer(use_idf=True,  
                        smooth_idf=True, min_df=3, max_features=2000,
                        ngram_range=(1,1), stop_words = stopwords,
                        tokenizer=lambda x:x,preprocessor=lambda x:x)
    tf_idf_data_smooth = tf_idf_vec_smooth.fit_transform(tokenized)

        
    result=pd.DataFrame(tf_idf_data_smooth.toarray(),columns=tf_idf_vec_smooth.get_feature_names())    

    print(result)




X_tfidf()

