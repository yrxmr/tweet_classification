import twint
import pandas as pd
from itertools import groupby
import re

# Utilisation du modèle Twint pour effectuer du scraping
def scrape(keywords, since, outfile):
    c = twint.Config()
    c.Search = keywords  # search keyword
    c.Since = since
    c.Store_csv = True
    c.Output = "./" + outfile
    c.Hide_output = True
    c.Count = True
    c.Stats = True
    c.Resume = 'resume.txt'
    twint.run.Search(c)

def get_tweets(infile, outfile):
    df = pd.read_csv(infile, usecols=["tweet"])
    df["tweet"].to_csv(outfile, index=False, header=False)

# Features
# Tous les features sont calculés au niveau de la phrase, donc au niveau d'un tweet
# Ils sont ensuite concaténés pour faire un dataframe comprenant l'ensemble des features

def parse_conllu(conllu_file):
    with open(conllu_file, "r") as file:
        lines = [re.sub("!|\?|\.{1,3}|,|:", "", line) for line in file]
    return [list(g) for k, g in groupby(lines, key=lambda x: x != '\n') if k]

def compute_features(sentences):
    percent_noun, percent_adj, percent_verb, lengths = [], [], [], []
    percent_inf, percent_plur, percent_sing = [], [], []
    percent_1st, percent_2nd, percent_3rd = [], [], []
    percent_punct, avg_word_len = [], []

    for sentence in sentences:
        length = len(sentence)
        lengths.append(length)

        noun_count = sum(1 for item in sentence if "NOUN" in item or "PROPN" in item)
        adj_count = sum(1 for item in sentence if "ADJ" in item)
        verb_count = sum(1 for item in sentence if "VERB" in item)
        
        percent_noun.append(noun_count / length if length else 0)
        percent_adj.append(adj_count / length if length else 0)
        percent_verb.append(verb_count / length if length else 0)

        inf_count = sum(1 for item in sentence if "VerbForm=Inf" in item)
        verb_total = sum(1 for item in sentence if "VerbForm" in item)
        plur_count = sum(1 for item in sentence if "VERB" in item and "Number=Plur" in item)
        sing_count = sum(1 for item in sentence if "VERB" in item and "Number=Sing" in item)
        first_pers_count = sum(1 for item in sentence if "Person=1" in item and "VERB" in item)
        second_pers_count = sum(1 for item in sentence if "Person=2" in item and "VERB" in item)
        third_pers_count = sum(1 for item in sentence if "Person=3" in item and "VERB" in item)

        percent_inf.append(inf_count / verb_total if verb_total else 0)
        percent_plur.append(plur_count / verb_total if verb_total else 0)
        percent_sing.append(sing_count / verb_total if verb_total else 0)
        percent_1st.append(first_pers_count / verb_total if verb_total else 0)
        percent_2nd.append(second_pers_count / verb_total if verb_total else 0)
        percent_3rd.append(third_pers_count / verb_total if verb_total else 0)

        punct_count = sum(1 for item in sentence if "punct" in item)
        percent_punct.append(punct_count / length if length else 0)
        
        word_lengths = [len(word) for item in sentence for word in re.findall(r"\w+", item) if word != "PUNCT"]
        avg_word_len.append(sum(word_lengths) / len(word_lengths) if word_lengths else 0)

    data = {
        'NOM%': percent_noun,
        'ADJ%': percent_adj,
        'VERB%': percent_verb,
        'NB LEMMES': lengths,
        'VINF%': percent_inf,
        'VPLUR%': percent_plur,
        'VSING%': percent_sing,
        '1ère PER%': percent_1st,
        '2ème PER%': percent_2nd,
        '3ème PER%': percent_3rd,
        'PUNCT%': percent_punct,
        'AVG LEN WORD': avg_word_len
    }

    return pd.DataFrame(data)

def save_to_csv(df, file):
    df.to_csv(file, index=False)

# Usage
# scrape('couvre-feu', '2021-01-1 15:55:00', 'AAtweets.csv')
# get_tweets("AAtweets.csv","outfile.txt")

conllu_file = 'conllu/file'  
output_csv = 'output.csv'  

sentences = parse_conllu(conllu_file)
features_df = compute_features(sentences)
save_to_csv(features_df, output_csv)
