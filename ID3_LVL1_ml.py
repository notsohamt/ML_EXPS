import pandas as pd
import numpy as np


file_path = r'C:\Users\dell\Desktop\sales_insights\random_dataset_20k_noise_1p0.csv' 
df = pd.read_csv(file_path)


df_encoded = df.copy()
for column in df_encoded.columns:
    if df_encoded[column].dtype == 'object':
        df_encoded[column] = df_encoded[column].astype('category').cat.codes

# Convert the target variable into binary classes 
df_encoded['Profit_Class'] = (df_encoded['Profit'].astype('category').cat.codes)

# Drop the target variable coloumn
df_encoded.drop(columns=['Profit'], inplace=True)

# Prepare data
X = df_encoded.drop(columns=['Profit_Class']).values
y = df_encoded['Profit_Class'].values
features = df_encoded.drop(columns=['Profit_Class']).columns

# Function to calculate entropy
def entropy(y):
    proportions = np.bincount(y) / len(y)
    return -np.sum(p * np.log2(p) for p in proportions if p > 0)

# information gain
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
def id3(X, y, features, depth=0, max_depth=1):
    # If all labels are the same, return that label
    if len(np.unique(y)) == 1:
        return y[0]
    
    # If no features are left or reached max depth, return the majority label
    if len(features) == 0 or depth >= max_depth:
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
        subtree = id3(subset_X, subset_y, new_features, depth + 1, max_depth)
        tree[features[best_feature]][value] = subtree
    
    return tree

# Function to print the tree
def print_tree(tree, level=0):
    if isinstance(tree, dict):
        for feature, branches in tree.items():
            print("  " * level + f"{feature}:")
            for value, subtree in branches.items():
                print("  " * (level + 1) + f"{value}:")
                print_tree(subtree, level + 2)
    else:
        print("  " * level + f"Leaf: {tree}")

# Build the ID3 tree up to level 1
tree = id3(X, y, list(features), max_depth=1)

# Print the tree
print("Decision Tree up to Level 1:")
print_tree(tree)
