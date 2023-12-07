import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data = pd.read_csv('Data.csv')

def cleanData(data):

    indexEmp = data[(data["employment_type"] != "FT")].index
    data.drop(indexEmp, inplace=True)
    data = data.drop(columns=["employment_type" , "salary", "salary_currency"])
    cutData = data.copy()
    #cutData["salary_in_usd"] = pd.cut(cutData["salary_in_usd"].values, bins= 5, labels=[0, 1, 2, 3, 4])
    
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

    X = cutData.iloc[:, 0]
    Y = cutData.iloc[:, 3]
    
    df = []

    for i, j in zip(X,Y):
        
        entry = [i,j]
        df.append(entry)

    #print(df)    
    dfArray = np.array(df)
    #dfArray = dfArray.T
    return dfArray

def k_means(df, k):
    # Randomly initialize centroids
    centroids = df[np.random.choice(len(df), k, replace=False)]

    for _ in range(100): #max number of iterations

        # Assign each data point to the nearest centroid
        labels = np.argmin(np.linalg.norm(df[:, np.newaxis] - centroids, axis=2), axis=1)

        # Update centroids based on the mean of the assigned data points
        new_centroids = np.array([df[labels == j].mean(axis=0) for j in range(k)])

        # Check for convergence
        if np.all(new_centroids == centroids):
            break

        centroids = new_centroids

    return labels, centroids

# Set the number of clusters (k)
k = 5

# Apply k-means algorithm
labels, centroids = k_means(cleanData(data), k)

data = cleanData(data)
#plt.scatter(dfArray[:,0], dfArray[:,1])
#plt.legend()
#plt.show()

# Visualize the results
plt.scatter(data[:,0], data[:,1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100, label='Centroids')
plt.legend()
plt.show()
