import twint
import pandas as pd
from itertools import groupby
import re


#utilisation du modèle Twint pour effectuer du scraping

def scrape (keywords, since, outfile):
        c = twint.Config()
        c.Search = keywords #search keyword
        c.Since = since
        c.Store_csv = True
        c.Output = "./" + outfile
        c.Hide_output = True
        c.Count = True
        c.Stats = True
        c.Resume = 'resume.txt'
        twint.run.Search(c)


def get_tweets(infile,outfile):
        col_list = ["tweet"]
        df = pd.read_csv(infile, usecols=col_list)
        rows = df["tweet"]
        with open(outfile, "w+") as f:
                for row in rows:
                        f.write(row+"\n")
                

#features
#tous les features sont calculés au niveau de la phrase, donc au niveau d'un tweet
#ils sont ensuite concaténés pour faire un dataframe comprenant l'ensemble des features
                        
def conllulemme(conllu):
    global dfs
    global df1
    global df2
    global df3
    global df4
    global lines
    dfs = []
    with open(conllu, "r") as file:
        lines = []
        for line in file:
            lines.append(line)
    for line in lines:
            replacement = re.sub("!|\?|\.{1,3}|,|:", "", line)
            line = replacement
            
    sentences = (list(g) for k,g in groupby(lines, key=lambda x: x != '\n') if k)
    percentagenoun = []
    percentageadj = []
    percentagev = []
    leng = []
    for sentence in sentences:
        x = len(sentence)
        a = 0
        b = 0
        c = 0
        for item in sentence:
            if "NOUN" in item or "PROPN" in item:
                a+=1
            if "ADJ" in item:
                b+=1
            if "VERB" in item:
                c+=1
        if a > 0:
            a = (a/x)
        else:
            a = "0"
        if b > 0:
            b = (b/x)
        else:
            b = "0"
        if c > 0:
            c = (c/x)
        else:
            c = "0"
        percentagenoun.append(a)
        percentageadj.append(b)
        percentagev.append(c)
        leng.append(x)
        df1 = pd.DataFrame(percentagenoun, columns=['NOM%'])
        df2 = pd.DataFrame(percentageadj, columns=['ADJ%'])
        df3 = pd.DataFrame(percentageadj, columns=['VERB%'])
        df4 = pd.DataFrame(leng, columns=['NB LEMMES'])
        dfs.extend([df1, df2, df3, df4])


        
def conlluverbes():
    global df5
    global df6
    global df7
    global df8
    global df9
    global df10
    sentences = (list(g) for k,g in groupby(lines, key=lambda x: x != '\n') if k)
    percentageinf = []
    percentageplur = []
    percentagesing = []
    firstpers = []
    secondpers = []
    thirdpers = []
    for sentence in sentences:
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        f = 0
        allv = 0
        for item in sentence:
            if "VerbForm=Inf" in item:
                    a+= 1
            if "VerbForm" in item:
                    allv+= 1
            if "VERB" in item and "Number=Plur" in item:
                    b+= 1
            if "VERB" in item and "Number=Sing" in item:
                    c+= 1
            if "Person=1" in item and "VERB" in item:
                    d+=1
            if "Person=2" in item and "VERB" in item:
                    e+=1
            if "Person=3" in item and "VERB" in item:
                    f+=1
                    
        if a > 0 and allv > 0:
            a = (a/allv)
            percentageinf.append(a)
        else:
            percentageinf.append("0")
        if b > 0 and allv > 0:
            b = (b/allv)
            percentageplur.append(b)
        else:
            percentageplur.append("0")
        if  c > 0 and allv > 0:
                c = (c/allv)
                percentagesing.append(c)
        else:
            percentagesing.append("0")
        if  d > 0 and allv > 0:
                d = (d/allv)
                firstpers.append(d)
        else:
            firstpers.append("0")
        if  e > 0 and allv > 0:
                e = (e/allv)
                secondpers.append(e)
        else:
            secondpers.append("0")
        if  f > 0 and allv > 0:
                f = (f/allv)
                thirdpers.append(e)
        else:
            thirdpers.append("0")
        df5 = pd.DataFrame(percentageinf, columns=['VINF%'])
        df6 = pd.DataFrame(percentageplur, columns=['VPLUR%'])
        df7 = pd.DataFrame(percentagesing, columns=['VSING%'])
        df8 = pd.DataFrame(firstpers, columns=['1ère PER%'])
        df9 = pd.DataFrame(secondpers, columns=['2ème PER%'])
        df10 = pd.DataFrame(thirdpers, columns=['3ème PER%'])
        dfs.extend([df5, df6, df7, df8, df9, df10])
        
        
def conllugraphie():
    global df11
    global df12
    global lines
    sentences = (list(g) for k,g in groupby(lines, key=lambda x: x != '\n') if k)
    a = 0
    b = 0
    percentagepunct= []
    length = []
    lenword = []
    for sentence in sentences:
        x = len(sentence)
        length.append(x)
        for item in sentence:
                if "punct" in item:
                    a+=1
                if a > 0 and x > 0:
                        a = (a/x)
                        percentagepunct.append(a)
                word = re.findall(r"\w+", item)
                if len(word) > 1 and word != "PUNCT":
                        b+=(len(word[1]))
        b = (b/x)
        lenword.append(b)


                        
                
                        
                        
    df11 = pd.DataFrame(percentagepunct, columns=['PUNCT%'])
    df12 = pd.DataFrame(lenword, columns=['AVG LEN WORD'])
    dfs.extend([df11, df12])

    

def csv(file):
        result = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12], axis=1).reindex(df1.index)
        result.to_csv(file)
   
#scrape('couvre-feu', '2021-01-1 15:55:00', 'AAtweets.csv')
#get_tweets("AAtweets.csv","outfile.txt")
conllulemme(fichierconllu)
conlluverbes()
conllugraphie()
csv(fichiercsv)
