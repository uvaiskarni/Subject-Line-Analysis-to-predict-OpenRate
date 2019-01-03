# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 11:01:50 2018

@author: Uvais Karni
"""
#import numpy as np
import pandas as pd
import nltk
import re
import string
import itertools
from nltk.util import ngrams
from nltk.corpus import stopwords
#from stemming.porter2 import stem
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
wnl = WordNetLemmatizer()


A = pd.read_csv('subject-line analyse input.csv', encoding = 'latin-1')

#A.columns

#corr=A.corr()

#A['TotalBounce']=A.TotalBase-A.TotalOpens
A['Open_Rate']=A.TotalOpens/A.TotalBase

median=A.Open_Rate.median()

A['Open_Rate']=A['Open_Rate']>median

    

A=A.drop(['Unnamed: 0','save', 'earth', 'life', 'aug', 'name', 'wate',
       'group', 'email', 'ignore', 'please', 'scrubbing', 'resulticks', 'team',
       'welcome', 'test', 'water', 'sdc', 'complete', 'solution', 'mcloud',
       'marketing', 'firstname', 'blasting', 'campaign', 'tube', 'youtube',
       'subject', 'testing', 'drive', 'growth', 'revenue', 'map', 'road',
       'deep', 'insight', 'worldclass', 'automation', 'businesses', 'digital',
       'generation', 'engagement', 'smart', 'india', 'hifirstname',
       'hifirstnamewelcome', 'link', 'analytics', 'yuo',
       'httpswwwyoutubecomwatchvzxwyiqwowkts'],axis=1)

#Stemming and tokenize the statements
def Stemming_Tokenize(A):
    
    temp1 = []
    temp2 = []
    temp3 = []
    temp4 = []
    unigrams=[]
    bigrams=[]
    checkname=[]
    checkfullname=[]
    checkusername=[]
    checkcustomerid=[]
    checkcity=[]
    checkemail=[]
    
    for text in A:

        checkname.append(Check_Name(text))
        checkfullname.append(Check_FullName(text))
        checkusername.append(Check_UserName(text))
        checkcustomerid.append(Check_CustomerID(text))
        checkcity.append(Check_City(text))
        checkemail.append(Check_Email(text))
        
        #converts all the SMS tex t to lower case
        text = text.lower()
        
        #removes any kind of punctuations from text
        text = re.sub(r'['+string.punctuation+']', r'', text)

        #remove stop words and stem words
        words = Stemming_StopWords(text)
        
        #count number of words in each sentence
        Count_Words = len(text.split())
        
        #Derive the unigrams
        unigrams.append(Unigram(words))
        
        #Derive the bigrams
        bigrams.append(Bigram(words))
        
        #list of SentKeyWords
        temp1.append(words)
        
        #Count of Words
        temp2.append(Count_Words)
    #finaly the words are again rejoined to make sentence after cleaning

    for row in temp1:
        sequ = ''
        Count_Keywords = len(row)
        temp3.append(Count_Keywords)
        for word in row:
            sequ = sequ + ' ' + word
        temp4.append(sequ)
        
#    substring=[]
#    substring = list(set(temp4))
#    for word in substring:
#        if word in text:
#            present=True
#        else:
#            present=False
#            list1.append(present)
#        checkwords.append(list1)
        
        
    return checkname,checkfullname,checkusername,checkcustomerid,checkcity,checkemail,temp1,temp2,temp3,temp4,unigrams,bigrams# temp1 list of keywords per stence ,temp2 list of stemmed sentence 

def Check_Name(text):
    substring = "[[Name]]"
    if substring in text:
        return True
    else:return False
    
def Check_FullName(text):
    substring = "[[Full Name]]"
    if substring in text:
        return True
    else:return False
    
def Check_UserName(text):
    substring = "[[User Name]]"
    if substring in text:
        return True
    else:return False
    
def Check_CustomerID(text):
    substring = "[[Customer ID]]"
    if substring in text:
        return True
    else:return False
    
def Check_City(text):
    substring = "[[City]]"
    if substring in text:
        return True
    else:return False

def Check_Email(text):
    substring = "[[Email ID]]"
    if substring in text:
        return True
    else:return False

def Gen_KeyWordsFreq(ListKeyWords):
    c=nltk.FreqDist(ListKeyWords)
    return c.most_common(50)

def Stemming_StopWords(text):
    
    return([stemmer.stem(wnl.lemmatize(word)) for word in text.split()if word not in stopwords.words('english')])

def Unigram(words):
    return(list(ngrams(words,1)))
    
def Bigram(words):
    return(list(ngrams(words,2)))

CheckName,CheckFullName,CheckUserName,CheckCustomerID,CheckCity,CheckEmail,SentKeyWords,Count_Words,Count_Keywords,Stemmedsent,Unigrams,Bigrams= Stemming_Tokenize(A.SubjectLine)
ListKeyWords= list(itertools.chain.from_iterable(SentKeyWords))
KeyWordsFreq=Gen_KeyWordsFreq(ListKeyWords)
ListKeyWords=list(set(ListKeyWords))    


