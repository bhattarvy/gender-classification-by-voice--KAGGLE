#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 04:34:15 2018

@author: arvy
"""

import pandas as pd
import numpy as np

data=pd.read_csv('/home/arvy/Documents/ML/datasets/voice.csv')

from sklearn.preprocessing import LabelEncoder
gle=LabelEncoder()
data.label=gle.fit_transform(data.label)


y=data.label
X=data.drop(['label'],axis=1)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42)


from sklearn.svm import SVC
model=SVC(C=1550)
model.fit(X_train,y_train)
print(model.score(X_test,  y_test))