import os
import pandas as pd
import numpy as np

os.chdir("C:\\Users\\user\\Documents\\Python\\Practises\\senti 1")

messages = pd.read_csv("Restaurant_Reviews.tsv",sep ='\t')


import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

ps = PorterStemmer()
corpus = []

for i in range(len(messages['Review'])):
    add = re.sub('[^a-zA-Z]',' ',messages['Review'][i])
    add = add.lower()
    add = add.split()
    add = [ps.stem(word) for word in add if word not in set(stopwords.words('english'))]
    message = ' '.join(add)
    corpus.append(message)
    

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

cv = CountVectorizer(max_features =500)
X = cv.fit_transform(corpus).toarray()

X = pd.DataFrame(X)

import pickle
CV = pickle.dump(cv,open('CV.pkl','wb'))

Y = messages['Liked']

Fullraw = pd.concat([X,Y],axis =1)

from sklearn.model_selection import train_test_split

Train,Test = train_test_split(Fullraw,test_size = 0.3,random_state =123)

Train_X = Train.drop(['Liked'],axis =1)
Train_Y = Train['Liked']
Test_X = Test.drop(['Liked'],axis =1)
Test_Y = Test['Liked']

M1 = MultinomialNB(alpha=0.9).fit(Train_X,Train_Y)

Test_pred = M1.predict(Test_X)

from sklearn.metrics import confusion_matrix

Con_Mat = confusion_matrix(Test_pred,Test_Y)
sum(np.diag(Con_Mat))/Test_Y.shape[0]*100

pickle.dump(M1,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))



    