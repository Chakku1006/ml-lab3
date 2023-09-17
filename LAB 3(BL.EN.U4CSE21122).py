#!/usr/bin/env python
# coding: utf-8

# In[18]:


#1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv(r"C:\Users\mamid\OneDrive\Desktop\MACHINE LEARNING LAB\extracted_features_charrec.csv")
grouped_data = data.groupby('class_name')
means = []
std_devs = []
for label, group in grouped_data:
    # Calculate the mean and standard deviation for each group
    # Replace 'numeric_column' with the column containing numerical data
    mean = np.mean(group['0'],axis=0)
    # Replace 'numeric_column' with the column containing numerical data
    std_dev = np.std(group['0'],axis=0)
    means.append((label, mean))
    std_devs.append((label, std_dev))
for label, mean in means:
    print(f"Label: {label}, Mean: {mean}")
for label, std_dev in std_devs:
    print(f"Label: {label}, Standard Deviation: {std_dev}")


# In[19]:


grouped_data = data.groupby('class_name')
# Initialize a dictionary to store the mean vectors for each class
class_means = {}
# Calculate and store the mean vector for each class
for label, group in grouped_data:
     # Replace '0' with the column containing numerical data
    mean = np.mean(group['0'], axis=0)
    class_means[label] = mean
# Get a list of unique class labels
unique_labels = list(class_means.keys())
# Calculate the distance between mean vectors for different classes
for i in range(len(unique_labels)):
    for j in range(i + 1, len(unique_labels)):
        label_1 = unique_labels[i]
        label_2 = unique_labels[j]
        mean_vector_1 = class_means[label_1]
        mean_vector_2 = class_means[label_2]
        distance = np.linalg.norm(mean_vector_1 - mean_vector_2)
        print(f"Distance between {label_1} and {label_2}: {distance}")


# In[20]:


feature_data = data['41']
# Define the number of bins (buckets) for the histogram
num_bins = 15
# Plot the histogram
plt.hist(feature_data, bins=num_bins, edgecolor='red')
plt.xlabel('Feature Values')
plt.ylabel('Frequency')
plt.title('Histogram of Feature Column')
plt.show()
# Calculating the mean and variance of the feature
mean = np.mean(feature_data)
variance = np.var(feature_data)
print(f"Mean: {mean}")
print(f"Variance: {variance}")


# In[21]:


#2
feature_data = data['class_name']
# Define the number of bins (buckets) for the histogram
num_bins = 34 
# Plot the histogram
plt.hist(feature_data, bins=num_bins, edgecolor='red')
plt.xlabel('Feature Values')
plt.ylabel('Frequency')
plt.title('Histogram of Feature Column')
plt.show()
# Calculate the mean and variance of the feature
mean = np.mean(feature_data)
variance = np.var(feature_data)
print(f"Mean: {mean}")
print(f"Variance: {variance}")


# In[22]:


#3
from scipy.spatial import distance
feature_vector1 = data['1']
feature_vector2 = data['2']
# Define a range of r values from 1 to 10
r_values = list(range(1, 8))
# Initialize a list to store the calculated distances for each r
distances = []
# Calculate Minkowski distances for each value of r
for r in r_values:
    distance_r = distance.minkowski(feature_vector1, feature_vector2, r)
    distances.append(distance_r)
# Plot the distances
plt.figure(figsize=(10, 6))
plt.plot(r_values, distances, marker='o', linestyle='--')
plt.xlabel('r (Minkowski Parameter)')
plt.ylabel('Minkowski Distance')
plt.title('Minkowski Distance vs. r')
plt.grid(True)
plt.show()


# In[23]:


#4
from sklearn.model_selection import train_test_split
# Specify the features and target variable
# Replace 'target_column' with your actual target column
X = data.drop(columns=['class_name']) 
# Replace 'target_column' with your actual target column
y = data['class_name']  
# Split the dataset into a train set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=39)

# Print the shapes of the train and test sets to verify the split
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


# In[24]:


#5
from sklearn.neighbors import KNeighborsClassifier

# Create a k-NN classifier with k=8
neigh = KNeighborsClassifier(n_neighbors=8)

# Train the classifier on the training data
neigh.fit(X_train, y_train)


# In[ ]:




