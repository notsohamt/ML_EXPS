import pandas as pd
import numpy as np

# Load the dataset from CSV file
file_path = r'C:\Users\dell\Desktop\sales_insights\random_dataset_20k_noise_1p0.csv'  # Update this path to the location of your CSV file
df = pd.read_csv(file_path)

# Convert the target variable into binary classes for simplicity
threshold = df['Profit'].median()
df['Profit_Class'] = (df['Profit'] > threshold).astype(int)

# Drop the original target column for classification
df.drop(columns=['Profit'], inplace=True)

# Function to calculate entropy
def entropy(y):
    proportions = np.bincount(y) / len(y)
    return -np.sum(p * np.log2(p) for p in proportions if p > 0)

# Function to calculate information gain
def information_gain(X, y, feature):
    # Calculate the entropy of the original data
    original_entropy = entropy(y)
    
    # Get unique values and their counts for the feature
    values, counts = np.unique(X[:, feature], return_counts=True)
    
    # Calculate the weighted average entropy for the splits
    weighted_entropy = 0
    for value, count in zip(values, counts):
        subset_y = y[X[:, feature] == value]
        weighted_entropy += (count / len(X)) * entropy(subset_y)
    
    # Information gain is the difference between the original entropy and the weighted entropy
    return original_entropy - weighted_entropy

# Function to find the best feature to split on
def best_feature_to_split(X, y):
    num_features = X.shape[1]
    best_gain = -1
    best_feature = -1
    
    for feature in range(num_features):
        gain = information_gain(X, y, feature)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature
    
    return best_feature

# Function to build the ID3 decision tree
def id3(X, y, features):
    # If all labels are the same, return that label
    if len(np.unique(y)) == 1:
        return y[0]
    
    # If no features are left, return the majority label
    if len(features) == 0:
        return np.bincount(y).argmax()
    
    # Find the best feature to split on
    best_feature = best_feature_to_split(X, y)
    
    # Create a tree node
    tree = {features[best_feature]: {}}
    
    # Remove the chosen feature from the list
    new_features = [f for i, f in enumerate(features) if i != best_feature]
    
    # Get unique values of the best feature and create branches
    for value in np.unique(X[:, best_feature]):
        subset_X = X[X[:, best_feature] == value]
        subset_y = y[X[:, best_feature] == value]
        subtree = id3(subset_X, subset_y, new_features)
        tree[features[best_feature]][value] = subtree
    
    return tree

# Prepare data
X = df.drop(columns=['Profit_Class']).values
y = df['Profit_Class'].values
features = df.drop(columns=['Profit_Class']).columns

# Build the ID3 tree
tree = id3(X, y, list(features))

# Print only the root node
if isinstance(tree, dict) and len(tree) > 0:
    root_node = list(tree.keys())[0]
    print(f"Root Node: {root_node}")
else:
    print("No root node found")
