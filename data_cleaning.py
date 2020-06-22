#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 17:46:57 2020

@author: Scott
"""

#path = '/content/drive/My Drive/Python scripts/Data Science/Summariser/'
path = ''


#%% Modules

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import nltk
import time
import re
import pickle
import sys

#%% General functions

def progressBar(n, tot):
    perc = int(100*n/tot)
    milli = round(100*n/tot,1)
    print('\r{}/{} ({}%) [{}{}]'.format(n,tot,milli,u'\u25A4'*perc, '_'*(100-perc)),end='')

#%% Loading the dataframe

load = 1

if load==0:
    df = pd.read_csv(path+'wikihowSep.csv')

    df.drop(columns=['overview','sectionLabel'],inplace=True)

    titles_to_drop = [
    'How to Be Well Read', # This is full of loads of specific titles
    ]

    for title in titles_to_drop:
        df = df[df['title']!=title]
        df.reset_index(drop=True, inplace=True)

    df = df.dropna().reset_index(drop=True)
    df['cleaned'] = 0
    start_index = 0

if load==1:
    try:
        df = pd.read_pickle(path+'df_split_transformed_June_GCP.pkl')
    except:
        df = pd.read_pickle(path+'df_split_transformed_June_01.pkl')
    start_index = df['cleaned'].sum()

#%% Cleaning functions

def replaceAbbr(text):
    #XX need spaces before and after every word when we do this
    abbrs = [
    ["don't","do not"], ["can't", "can not"], ["won't", "will not"], ["shan't", "shall not"],
    ["dont","do not"], ["cant", "can not"], ["wont", "will not"], ["shant", "shall not"],

    ["i'll", "i will"], ["you'll", "you will"], ["youll", "you will"],
    ["we'll", "we will"], ["he'll", "he will"], ["she'll", "she will"],

    ["i'm", "i am"], ["im", "i am"], ["you're", "you are"], ["youre", "you are"],
    ["we're", "we are"],

    ["i've", "i have"], ["ive", "i have"], ["you've", "you have"], ["youve", "you have"],
    ["we've", "we have"]
    ]

    for ab, full in abbrs:
        text = re.sub("( {} )".format(ab), " {} ".format(full), text)
    return text

def cleanText(text):
    # Include \n\n = paragraph?
    # Include small and large numbers?
    if '__start__' in text or '__end__' in text: return text
    text = replaceAbbr(text)
    text = re.sub(r'(\.\s)', ' __fs__ ', text)
    text = re.sub(r'[,]', ' __cm__ ', text)
    text = re.sub(r'[/]', ' or ', text)
    text = re.sub(r'[\n-]', ' ', text)
    text = re.sub(r'[^a-zA-Z\d_\s]', '', text)
    text = text.lower()
    text = text.strip(' ')
    text = '__start__ ' + text + ' __end__'
    return text

#%% Perform cleaning

cleanLimit = len(df)
start_index = df['cleaned'].sum()

for irow in range(start_index, cleanLimit):
    df.loc[irow, 'text']     = cleanText(df.loc[irow, 'text'])
    df.loc[irow, 'headline'] = cleanText(df.loc[irow, 'headline'])
    df.loc[irow, 'title']    = cleanText(df.loc[irow, 'title'])
    df.loc[irow, 'cleaned'] = 1
    if irow%10==0:
        progressBar(irow,cleanLimit)
    if irow%100==0:
        df.to_pickle(path+'df_split_transformed_June_GCP.pkl')
progressBar(cleanLimit,cleanLimit)
print('Done')
df.to_pickle(path+'df_split_transformed_June_GCP.pkl')
























