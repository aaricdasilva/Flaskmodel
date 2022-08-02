import numpy as np
import pandas as pd
import pickle
from sklearn import datasets
from sklearn import linear_model
data = pd.read_csv(r'D:\intern\fish_data\fish.csv')
from sklearn.preprocessing import StandardScaler

# define the data/predictors as the pre-set feature names
scalar = StandardScaler()
scalar.fit(data.drop(columns = ['Species']))
df = scalar.transform(data.drop(columns = ['Species']))
# categorise the target variables
target = data['Species'].astype('category').cat.codes

lm = linear_model.LogisticRegression()
model = lm.fit(df,target)

def pred(x):
    if type(x)==list:
        pred_data = [x]
    else:
        pred_data = x
    preds = model.predict(pred_data)
    return preds


with open('mdl.pkl', 'wb') as f:
    pickle.dump(model,f)
    print('pickle completed')

with open('data.pkl', 'wb') as g:
    pickle.dump(data.drop(columns = ['Species']),g)
    print('pickle completed')