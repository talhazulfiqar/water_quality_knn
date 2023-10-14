import numpy as np
import pandas as pd

df = pd.read_csv('water_potability.csv').dropna() #Dropna function is used to remove all rows with missing values

#Splitting the data into 90% training and 10% testing
total_entries = len(df)
train_test_split = int(0.9 * total_entries)
train_data = df[:train_test_split].to_numpy()[:,:9]
train_labels = df[:train_test_split].to_numpy()[:,9].astype(int)
test_data = df[train_test_split:].to_numpy()[:,:9]
test_labels = df[train_test_split:].to_numpy()[:,9].astype(int)

class kNN:
    def __init__(self, k):
        self.k = k #Number of neighbours checked to calculate prediction
    def train(self, data, labels):
        #Training is simple, we just store the data in memory
        self.train_data = data
        self.labels = labels
    def predict(self, point):
        total_points = self.train_data.shape[0] #First value in shape is the total number of entries
        all_distances = [np.sum(np.abs(p - point)) for p in self.train_data]
        smallest_k_values_indexes = np.argpartition(all_distances, self.k)[:self.k] #Find the indexes of the k neighbours with smallest distance to the given point
        closest_labels = self.labels[smallest_k_values_indexes]
        voted_result = np.bincount(closest_labels).argmax() #Count which label is more
        return voted_result

network = kNN(12)
network.train(train_data, train_labels)


# Testing accuracy
correct = 0
total = test_labels.shape[0]
predictions = [network.predict(n) for n in test_data]
correct = np.count_nonzero(predictions == test_labels)

print("{}%".format((correct / total) * 100))