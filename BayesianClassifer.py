import numpy as np
import time as t
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data = pd.read_csv('Data.csv')
X = []
Y = []

#process and clean data. Lable columns and classes. 
def cleanData(data):

    indexEmp = data[(data["employment_type"] != "FT")].index
    data.drop(indexEmp, inplace=True)
    data = data.drop(columns=["employment_type" , "salary", "salary_currency"])
    cutData = data.copy()
    cutData["salary_in_usd"] = pd.cut(cutData["salary_in_usd"].values, bins= 5, labels=[0, 1, 2, 3, 4])
    
    #classes
    year = LabelEncoder()
    exp_lvl = LabelEncoder()
    job_title = LabelEncoder()
    country = LabelEncoder()
    comp_size = LabelEncoder()

    #cutData = cutData.iloc[:,:].values

    #fit encoded variables to specific columns
    cutData.iloc[:, 0] = year.fit_transform(cutData.iloc[:,0])
    cutData.iloc[:, 1] = exp_lvl.fit_transform(cutData.iloc[:, 1])
    cutData.iloc[:, 2] = job_title.fit_transform(cutData.iloc[:, 2])
    cutData.iloc[:, 4] = country.fit_transform(cutData.iloc[:, 4])
    cutData.iloc[:, 5] = comp_size.fit_transform(cutData.iloc[:, 5])
    #year = 0, experiance level = 1, job title = 2, country = 4, company size = 5
    #salary = 3

    loList = []
    mlList = []
    miList = []
    mhList = []
    hiList= []
    for i in range(len(data)):
        if(cutData.iloc[i, 3] == 0): loList.append(data.iloc[i, 3])
        elif(cutData.iloc[i, 3] == 1): mlList.append(data.iloc[i, 3])
        elif(cutData.iloc[i, 3] == 2): miList.append(data.iloc[i, 3])
        elif(cutData.iloc[i, 3] == 3): mhList.append(data.iloc[i, 3])
        elif(cutData.iloc[i, 3] == 4): hiList.append(data.iloc[i, 3])

    X = cutData.iloc[:, 3]
    Y = cutData.iloc[:,:]
    #print(Y)
    
    return 



def calculatePrior(df, Y):
    classes = sorted(list(df[Y].unique()))
    prior = []
    for i in classes:
        prior.append(len(df[df[Y] == i])/len(df))
    return prior

def calculateLikelihoodGuassian(df, featName, featValue, Y, label):
    feat = list(df.columns)
    print(feat)
    df= df[df[Y] == label]
    mean = df[featName].mean()
    std = df[featName].std()

    pXGivenY = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((featValue - mean)**2 / (2 * std **2)))
    return pXGivenY

def naiveBayesGuassian(df, X, Y):
    features = list(df)

    prior = calculatePrior(df, Y)

    YPrediction = []

    for x in X:
        lables = sorted(list(df[Y].unique()))
        likelihood = [1] * len(lables)
        for j in range(len(lables)):
            for i in range(len(features)):
                likelihood[j] = calculateLikelihoodGuassian(df, features[i], x[i], Y[i], lables[j])

    postProbability = [1] * len(lables)
    for j in range(len(lables)):
        postProbability[j]= likelihood[j] * prior[j]

    YPrediction.append(np.argmax(postProbability))

    return np.array(YPrediction)


#Split data into training-set and test-set
from sklearn.model_selection import train_test_split
train, test = train_test_split(cleanData(data), test_size=.2, random_state=42)

#yPred = naiveBayesGuassian(cleanData(data), xTest, 4)

from sklearn.metrics import f1_score

cleanData(data)

#print("data")