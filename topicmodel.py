#Importing all the required packages

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import re
import glob
import codecs	

#Extracting the files from specified path and loading the data

path = "C:\\Users\\PRASANNA\\datasets\\bbc small\\*.txt"
files = glob.glob(path)
dataset = []
for file in files:
    with codecs.open(file, encoding="utf8", errors='ignore') as f:
        dataset.append(" ".join(f.readlines()))

#Creating a bag of words that can be supplied to the TF-IDF vectorizer inorder to create TF-IDF Matrix A

vectorizer = CountVectorizer(min_df = 1)
words = []
for x in dataset:
    try:
        vectorizer.fit_transform(x.split())
        words.append(vectorizer.get_feature_names())
    except:
        print "\nException"
lemmatizer = WordNetLemmatizer()
bagOfWords = [ " ".join([lemmatizer.lemmatize(x) for x in word_list ]) for word_list in words ]
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, strip_accents="unicode",\
                                stop_words="english",use_idf=True)
tfidf = vectorizer.fit_transform(bagOfWords)

#Hyper parameters that describe the number of topics of interest and the top N words under each topic

number_of_topics = 5
top_nwords = 10

#Performing Non-negative Matrix Factorisation over the constructed TF-IDF Matrix A

nmf = NMF(n_components=5, random_state=1).fit(tfidf)
W = nmf.components_
H = nmf.fit_transform(tfidf)
topics = [[]] * number_of_topics
feature_names = vectorizer.get_feature_names()
for i in range(len(W)):
    topics[i] = [ feature_names[j] for j in np.argsort(W[i,:])[:-top_nwords - 1:-1] ]
	
#Finally, the results :)

for i in range(number_of_topics):
    print("Topic "+str(i)+": ")
    print(",".join(topics[i]))

"""
Topic 0:
game,player,club,team,goal,league,cup,season,manager,win
Topic 1: 
year,market,rate,growth,price,profit,analyst,economy,rise,month
Topic 2: 
virus,program,firm,user,computer,software,site,mail,website,pc
Topic 3: 
minister,tory,party,government,election,mr,plan,labour,say,leader
Topic 4: 
technology,device,mobile,phone,digital,video,service,gadget,million,music
"""