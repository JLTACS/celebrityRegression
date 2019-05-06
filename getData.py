import pandas as pd
import numpy as np 
import random

def prepareData():
    df = pd.read_csv("list_attr_celeba.csv", sep=',',header=0, names=None)
    df_filter = df.drop(columns = ['image_id','Blurry','Mouth_Slightly_Open','Smiling','Wearing_Hat','Wearing_Lipstick','Wearing_Necklace','Wearing_Necktie','Heavy_Makeup'])
    df_filter.replace({-1:0},inplace = True)

    rt = df_filter.pop('Attractive')
    rows, cols = df_filter.shape
    df_filter.insert(cols   ,'Attractive',rt)
    return df_filter

print(prepareData())

def separateTrainTest(data, train):
    df_train = pd.DataFrame(columns = data.columns)
    df_test = pd.DataFrame(columns = data.columns)
    train = train/100.0
   
    msk = np.random.rand(len(data)) < train
    df_train = data[msk]
    df_test = data[~msk]
    
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    with open("mi_train.csv",mode='w',newline='') as f:
        df_train.to_csv(f)
    with open("mi_test.csv",mode='w',newline='') as f:
        df_test.to_csv(f)
    return df_train,df_test

def separateDataXY(train,test):
    xtrain = train.iloc[:,:len(train.columns)-1].to_numpy(dtype = np.float32)
    ytrain = train.iloc[:,-1].to_numpy(dtype = np.float32)

    xtest = test.iloc[:,:len(test.columns)-1].to_numpy(dtype = np.float32)
    ytest = test.iloc[:,-1].to_numpy(dtype = np.float32)

    return xtrain,ytrain,xtest,ytest
