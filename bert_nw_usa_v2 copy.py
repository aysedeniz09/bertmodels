from bertopic import BERTopic
import pandas as pd
import csv
import re
import string
import datetime
import scipy
import numpy
from scipy import sparse
import sys   
import unicodedata
import nltk 
import numpy as np   
import hdbscan
import time     
from scipy.sparse import csr_matrix, csc_matrix 

from flair.embeddings import TransformerDocumentEmbeddings
roberta = TransformerDocumentEmbeddings('roberta-base')

nwdata = pd.read_csv(r'/home/adl6244/NewswhipUSA/Bert_TM/NW_USA_Paragraph_Master_052722.csv', encoding = "ISO-8859-1", engine='python')
nwdata.dropna(subset=['text'])
nan_value = float("NaN")
nwdata.replace("", nan_value, inplace=True)
nwdata.dropna(subset = ["text"], inplace=True)
nwdata = nwdata[pd.notnull(nwdata['date'])]
nwdata['date'] = pd.to_datetime(nwdata["date"], utc=True)
#nwdata.info()
#nwdata.head()

stop_words = pd.read_csv(r'/home/adl6244/NewswhipUSA/Bert_TM/stop_words.csv', encoding = "ISO-8859-1", engine='python')
stop_words = stop_words.text.to_list()

def text_clean(x):

    ### Light
    x = x.lower() # lowercase everything
    x = x.encode('ascii', 'ignore').decode()  # remove unicode characters
    x = re.sub(r'https*\S+', ' ', x) # remove links
    x = re.sub(r'http*\S+', ' ', x)
    # cleaning up text
    x = re.sub(r'\'\w+', '', x) 
    x = re.sub(r'\w*\d+\w*', '', x)
    x = re.sub(r'\s{2,}', ' ', x)
    x = re.sub(r'\s[^\w\s]\s', '', x)
    
    ### Heavy
    x = ' '.join([word for word in x.split(' ') if word not in stop_words])
    x = re.sub(r'@\S', '', x)
    x = re.sub(r'#\S+', ' ', x)
    x = re.sub('[%s]' % re.escape(string.punctuation), ' ', x)
    # remove single letters and numbers surrounded by space
    x = re.sub(r'\s[a-z]\s|\s[0-9]\s', ' ', x)

    return x

nwdata['cleaned_text'] = nwdata.text.apply(text_clean)

timestamps = nwdata.date.to_list()
nwtext = nwdata.cleaned_text.to_list()

start_time = time.time()
topic_model = BERTopic(embedding_model=roberta, nr_topics="auto", calculate_probabilities = True).fit(nwtext)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
docs = topic_model.get_representative_docs()
freq = topic_model.get_topic_info()
freq.to_csv("nw_usa_052922.csv")
print("--- %s seconds ---" % (time.time() - start_time))

doc_panda = pd.DataFrame(list(docs.items()),columns = ['Topic','Docs']) 
doc_panda.to_csv("nw_covid_representative_docs_052922.csv")

topic_model.save("modelcovidroberta_052922") 

topics, probs = topic_model.fit_transform(nwtext)
topics_over_time = topic_model.topics_over_time(nwtext, topics, timestamps, nr_bins=20)
topics_over_time.to_csv("nw_covid_overtime_052922.csv")

probs = hdbscan.all_points_membership_vectors(topic_model.hdbscan_model)
probs = topic_model._map_probabilities(probs, original_topics=True)

df = pd.DataFrame(probs)
df.to_csv("nw_covid_probs_052922.csv")

docs = topic_model.get_representative_docs()
freq = topic_model.get_topic_info()
freq.to_csv("nw_freqpostprobs_052922.csv")
print("--- %s seconds ---" % (time.time() - start_time))

doc_panda = pd.DataFrame(list(docs.items()),columns = ['Topic','Docs']) 
doc_panda.to_csv("nw_covid_representative_docs_postprob_052922.csv")

topic_model.save("modelcovidroberta_v2_052922") 
 
###third model
start_time = time.time()
topic_model = BERTopic(embedding_model=roberta, nr_topics=50, calculate_probabilities = True).fit(nwtext)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
docs = topic_model.get_representative_docs()
freq = topic_model.get_topic_info()
freq.to_csv("nw_covid_k50_052922.csv")
print("--- %s seconds ---" % (time.time() - start_time))

doc_panda = pd.DataFrame(list(docs.items()),columns = ['Topic','Docs']) 
doc_panda.to_csv("nw_covid_representative_docs_k50_052922.csv")

topic_model.save("modelcovidroberta_k50_052922") 

topics, probs = topic_model.fit_transform(nwtext)
topics_over_time = topic_model.topics_over_time(nwtext, topics, timestamps, nr_bins=20)
topics_over_time.to_csv("nw_covid_overtime_k50_052922.csv")

probs = hdbscan.all_points_membership_vectors(topic_model.hdbscan_model)
probs = topic_model._map_probabilities(probs, original_topics=True)

df = pd.DataFrame(probs)
df.to_csv("nw_covid_probs_k50_052922.csv")
