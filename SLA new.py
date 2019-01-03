# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 16:14:53 2018

@author: Uvais Karni
"""


import pandas as pd 
import language_processing as lp
import itertools
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
#from scipy.sparse import csc_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn_pandas import DataFrameMapper
import sklearn
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from wordcloud import WordCloud, STOPWORDS 

def DataPreprocessing():
    #-------importing the csv file and storing it as a new df 
    A = pd.read_excel('expdata.xlsx', encoding = 'latin-1')

    #-----------Tokenizing the subjectline and storing in a separate column 
    A['Tokens']= lp.stemmingTokenize(A.EmailSubject)

    #--------------calculating the count of words and storing in a separate column
    A['wordcount']=list(map(lp.countwords,A.EmailSubject))

    #-----------------calculating open rate and naming it as targetopenrate
    A['target_open_rate']=A['Email_Opened']/A['EmailDelivered']

    #--------------------Area to operate :
    list_words=list(itertools.chain.from_iterable(A.loc[:,'Tokens']))
    set_words=set(list_words)
    len(set_words)
    #f=lp.getWordFrequency(list_words)
    #word_features = list(f.keys())[:200]
    #>>> Work on getting only the top n frequent words 

    #-----------------------creating word vectors 



    mapper = DataFrameMapper([
            ('wordcount',None),('EmailSubject', TfidfVectorizer(
                    sublinear_tf=True,
                    strip_accents='unicode',
                    analyzer='word',
                    token_pattern=r'\w{1,}',
                    stop_words='english',
                    ngram_range=(1, 1),
                    max_features=2000)),
                    ])
    X=mapper.fit_transform(A.copy())
    #X,y split 
    y=A['target_open_rate']
    return(X,y,A)

#--------------------end of DataPreprocessing function












def RecommendedWords():
    #getting the 75th percentile of the openrate 
    open_rate_75th = np.percentile(A['target_open_rate'],75)
    #Exctracting tokens that lie in the 4th quartile 
    #And combining them to a single list 
    list_4thQ=list(itertools.chain.from_iterable(A.loc[A['target_open_rate']>open_rate_75th,'Tokens']))
    #&&&&
    f_4thQ=lp.getWordFrequency(list_4thQ)
    #Forming the recommended Words list:
    #converting dictionary to df:
    recommended_words_df=pd.DataFrame(list(f_4thQ.keys()))
    recommended_words_df['frequency']=list(f_4thQ.values())
    #find a better way to convert dict to dataframe. Both Keys and values 
    recommended_words_df.columns=['words','frequency']
    #calculating the 75th percentile 
    f_75th = np.percentile(recommended_words_df['frequency'],75)
    #Finally creating the recommended words list 
    recommended_words_list=recommended_words_df.loc[recommended_words_df['frequency']>f_75th,'words']
    return(recommended_words_list,list_4thQ)
    
#-----------------------End of Recommended Words function 

    
    
def NotRecommendedWords():
    #getting the 75th percentile of the openrate 
    open_rate_25th = np.percentile(A['target_open_rate'],25)
    #Exctracting tokens that lie in the 1st quartile 
    #And combining them to a single list 
    list_1stQ=list(itertools.chain.from_iterable(A.loc[A['target_open_rate']<=open_rate_25th,'Tokens']))
    #$$$$$$$$
    f_1stQ=lp.getWordFrequency(list_1stQ)
    #Forming the recommended Words list:
    #converting dictionary to df:
    #converting dictionary to df:
    not_recommended_words_df=pd.DataFrame(list(f_1stQ.keys()))
    not_recommended_words_df['frequency']=list(f_1stQ.values())
    #find a better way to convert dict to dataframe. Both Keys and values 
    not_recommended_words_df.columns=['words','frequency']
    #calculating the 75th percentile 
    f_75th_not = np.percentile(not_recommended_words_df['frequency'],75)
    #Finally creating the recommended words list 
    not_recommended_words_list=not_recommended_words_df.loc[not_recommended_words_df['frequency']>f_75th_not,'words']
    return(not_recommended_words_list,list_1stQ)
    
    


def modeling(X_train,y_train,X_test,y_test):
    #XGB regressor
    model=[]
    meanSquaredError=[]
    modelXGBR = xgb.XGBRegressor(n_estimators=1000,max_depth=3)
    modelXGBR.fit(X_train,y_train)
    print(modelXGBR)
    #y_pred1=modelXGBR.predict(X_train)
    y_pred=modelXGBR.predict(X_test)
    meanSquaredError.append(sklearn.metrics.mean_squared_error(y_test,y_pred))
    model.append(modelXGBR)
    #    sklearn.metrics.r2_score(y_train, y_pred1)
    #    sklearn.metrics.r2_score(y_test, y_pred)    
    #    plt.scatter(y_test,y_pred)
    #CatBoostRegressor 
    model_cat = CatBoostRegressor(iterations=2000,
                                  learning_rate=0.06,
                                  depth=3,
                                  l2_leaf_reg=4,
                                  border_count=15,
                                  loss_function='RMSE',
                                  verbose=200)

    model_cat.fit(X_train,y_train)
    model.append(model_cat)
    #y_pred1_cat=model_cat.predict(X_train)
    y_pred_cat=model_cat.predict(X_test)
    meanSquaredError.append(sklearn.metrics.mean_squared_error(y_test,y_pred_cat))    
    minIndex=meanSquaredError.index(min(meanSquaredError))
    return(model[minIndex])        
    
#----------------------end of DataPreprocessing function


#Function for word cloud 

def cloudGenerator(list1):
    stopwords = set(STOPWORDS)
    comment_words = ' '
    for words in list1: 
        comment_words = comment_words + words + ' '

    wordcloud = WordCloud(width = 800, height = 800, 
                          background_color ='white', 
                          stopwords = stopwords, 
                          min_font_size = 10).generate(comment_words)
    return(wordcloud)
    

#-----------------------




X,y,A=DataPreprocessing()
#Train test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=4)
finalModel=modeling(X_train,y_train,X_test,y_test)
y_pred=finalModel.predict(X_test)
sklearn.metrics.mean_squared_error(y_test,y_pred)
ls_recommended,list_4thQ=RecommendedWords()
ls_NotRecommendedWords,list_1stQ=NotRecommendedWords()
# 1. Check the functions 
# 2. Generate word cloud 

#    sklearn.metrics.r2_score(y_train, y_pred1_cat)
#    sklearn.metrics.r2_score(y_test, y_pred_cat) 
#    plt.scatter(y_test,y_pred_cat)





# plot the WordCloud image for recommended words                         

wordcloud=cloudGenerator(list_4thQ)
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 





# plot the WordCloud image for not recommended words                    
wordcloud1=cloudGenerator(list_1stQ)
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud1) 
plt.axis("off") 
plt.tight_layout(pad = 0)   
plt.show() 
