import time as t
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Assuming df is your DataFrame
# Replace 'Column_Name' with the name of your column
df = pd.read_csv('Data.csv')

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

five = [50.2, 53.5, 51.6, 50.4, 48.9]
two = [86.4, 84.6, 84.9, 87.3, 84.9]
ten = [26.9, 25.5, 25.7, 23.2, 22.4]

#print(np.std(ten))

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create an empty list to store the WCSS values for different k
wcss = []

# Assuming your data is stored in multiple columns in the DataFrame
X = cleanData(df).values  # Assuming df contains your data

# Calculate WCSS for different values of k
for i in range(1, 11):  # Testing k from 1 to 10 clusters
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # Inertia is the WCSS value

# Plot the WCSS values against the number of clusters (k)
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()
