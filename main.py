# imports section
#basic imports 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

#advanced imports 
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
# https://www.guru99.com/python-counter-collections-example.html 
# counter holds objects and gives a count of those objects within itself 
from collections import Counter

# cool interactive website that actively shows the algorithm that is at work in this project
# https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/

# arrays to hold random x and y values
x_values = np.array([]) # array x vals
y_values = np.array([]) # array of y vals

# go through 100 points and create a random x and y val into their arrays
for x in range(0, 100):
    x_values = np.append(arr=x_values, values=random.randint(1, 250))
    y_values = np.append(arr=y_values, values=random.randint(1, 250))


df = pd.DataFrame(x_values, y_values)
# print(df)
# push the dataframe to a csv for use in the DBSCAN
df.to_csv("points.csv", header=False)

# make the data a CSV file
data = pd.read_csv("points.csv", sep=',', names=["X", "Y"])

# plots the data onto a graph -- not sure of why using the underscore as the name, but wasnt sure if this needed because of the model.labels_ later 
_ = plt.plot(data["X"], data["Y"], marker=".", linewidth=0, color='#128128')
# plt.show()


# covert the data into an array
dbscan_data = data[["X", "Y"]]
dbscan_data = dbscan_data.values.astype('float32', copy=False)

# normalize the data ?
# this scaler makes the data somehow fit into a scale of 1 - then the epsilon is under that one
dbscan_data_scaler = StandardScaler().fit(dbscan_data)
dbscan_data = dbscan_data_scaler.transform(dbscan_data)

# create the model to run the Density Scan across -- the circle gif we looked at
# eps is the circle (epsilon)
# min samples is the minimum number of points within the epsilon to be considered a cluster
model = DBSCAN(eps=0.25, min_samples=3, metric='euclidean')
model.fit(dbscan_data)
# the model makes labels on each point -1 means it an outlier, then the rest numbers mean they are part of a cluster
# to be an outlier means that there are no other points within the epsilon surrounding that point

# data frame that holds the clusters defined either in the cluster or as an outlier
outlier_df = data[model.labels_ == -1] # -1 are outliers 
cluster_df = data[model.labels_ != -1] # anything else are numbers of points inside esp 

# creating different colors for each cluster
# this im not sure sure on how the line 58 and 59 work, line 60 is pretty straight forward 
colors = model.labels_
colors_clusters = colors[colors != -1]
colors_outliers = 'white'

# variable clusters is an object that holds each differnt label, and a count of how many there are 
clusters = Counter(model.labels_)

# ex this will be something like -1: 48, 1: 4, 2: 45 ... this means there is a label -1 with 48 points, a label 1 with 4 points ... 
print("Clusters : ", clusters)

# prints 5 examples (head is the first 5 of the df) of outliers -- points that were labeled -1
print(data[model.labels_ == -1].head())

# prints the total number of clusters (by our def of model = DBSCAN)
print("Number of clusters = {} ".format(len(clusters)-1))

# not completely sure how all this section works 
fig = plt.figure()  # creates a figure 
ax = fig.add_axes([.1, .1, 1, 1]) # ??? 
ax.scatter(cluster_df["X"], cluster_df["Y"], c=colors_clusters, edgecolors='black', s=50)   # puts a scatter of clusters on the figure 
ax.scatter(outlier_df["X"], outlier_df["Y"], c=colors_outliers, edgecolors='black', s=50)   # puts a scatter of the outliers on the figure 
plt.show()  # puts the figure on screen 
