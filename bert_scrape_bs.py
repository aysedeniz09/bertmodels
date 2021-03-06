##run a sample and see what you need to add to blacklist list

import requests
from bs4 import BeautifulSoup
from pprint import pprint
import pandas as pd
import csv

dfmain = pd.read_csv (r'spike_train.csv', encoding = "ISO-8859-1", engine='python')
list_of_urls = dfmain['link'].tolist()

rows = []
for link in list_of_urls:
    try:
        r = requests.get(link)
        soup = BeautifulSoup(r.content, "html.parser")
        text = soup.find_all(text=True)
        
        output = ''
        blacklist = [
            '[document]',
            'noscript',
            'header',
            'html',
            'meta',
            'head', 
            'input',
            'script',
    # there may be more elements you don't want, such as "style", etc.
        ]
        for t in text:
            if t.parent.name not in blacklist:
                output += '{} '.format(t)
        row = {'link':link,
               'soup':soup,
               'text':text,
              'output':output}
        
        rows.append(row)
    except Exception as e:
        row = {'link':link,
        'soup':'N/A',
        'text':'N/A',
        'output':'N/A'}
        
        rows.append(row)
        
df = pd.DataFrame(rows)
     
dfmaster = dfmain.merge(df, left_on='link', right_on='link')

dfmaster.to_csv('bert_train_scraped.csv')
