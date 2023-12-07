import time as t
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data = pd.read_csv('Data.csv')

#process and clean data. Lable columns and classes. 
def cleanData(data):

    indexEmp = data[(data["employment_type"] != "FT")].index
    data.drop(indexEmp, inplace=True)
    data = data.drop(columns=["employment_type" , "salary", "salary_currency"])
    cutData = data.copy()
    #cutData["salary_in_usd"] = pd.cut(cutData["salary_in_usd"].values, bins= 2, labels=[0, 1])
    cutData["salary_in_usd"] = pd.cut(cutData["salary_in_usd"].values, bins= 5, labels=[0, 1, 2, 3, 4])
    #cutData["salary_in_usd"] = pd.cut(cutData["salary_in_usd"].values, bins= 10, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    #features
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


    return cutData


def calculatePrior(df, Y):
    classes = sorted(list(df[Y].unique()))
    prior = []
    for i in classes:
        prior.append(len(df[df[Y] == i])/len(df))
    return prior


def calculateLikelihoodGaussian(df, featName, featValue, Y, label):
    df_label = df[df[Y] == label]
    
    if len(df_label) == 0:
        return 0  # Return 0 probability if the label doesn't exist in the data

    mean = df_label[featName].mean()
    std = df_label[featName].std()

    pseudocount = 1

    if std == 0:
        std += pseudocount  # Adding pseudocount to prevent division by zero
    
    pXGivenY = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((featValue - mean)**2 / (2 * std**2)))
    return pXGivenY



def naiveBayesGaussian(df, X, Y):
    features = list(X.columns)
    print("features: " ,features)
    print("class: 'salary_in_usd'")
    timeStart = t.time()
    prior = calculatePrior(df, Y)
    #print("Prior: " ,prior)
    YPrediction = []

    print("training on data in progress...")

    for x in X.values:
        
        labels = sorted(list(df[Y].unique()))
        
        likelihood = [1] * len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] = calculateLikelihoodGaussian(df, features[i], x[i], Y, labels[j])

            postProbability = [1] * len(labels)
            for j in range(len(labels)):
                postProbability[j] = likelihood[j] * prior[j]

            YPrediction.append(np.argmax(postProbability))

    timeEnd = t.time()
    duration = timeEnd - timeStart
    duration = "{:.1f}".format(duration)
    print(f"training done, took {duration}s")
    return np.array(YPrediction)

df = cleanData(data)

#Split data into training-set and test-set
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=.2, random_state=12) #1, 12, 23, 36, 50


yTest = test.iloc[:, 3].values
#yTest = np.array(yTest)
xTest = test.drop(columns=["salary_in_usd"])
xtest = xTest.iloc[:,:].values
yPred = naiveBayesGaussian(train, xTest, "salary_in_usd")

accuracy = 0

for i,j in zip(yPred,yTest):
    #print(i," ," ,j)
    if(i == j):
        accuracy += 1

print(len(yPred), ", " , len(yTest))
print("Accuracy = ",accuracy/len(yTest) ,",",  accuracy, " of ", len(yTest))