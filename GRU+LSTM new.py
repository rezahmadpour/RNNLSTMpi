#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import torch
import torch.nn as nn
import csv
from torch.utils.data import TensorDataset, DataLoader

DATA_FOLDER = 'D:/Thesis 2/data'
REPORT_FOLDER = 'D:/Thesis 2/report'
DATA_FILE = os.path.join(DATA_FOLDER, 'pi1.txt')
REPORT_FILE = os.path.join(REPORT_FOLDER, 'model_results.csv')

# Hyperparameters
NUM_CLASSES = 10
NUM_EPOCHS = 1
BATCH_SIZE = 1
LEARNING_RATE = 0.001

INPUT_SIZE = 10
SEQUENCE_LENGTH = 1
NUM_LAYERS = 2


def log(message):
    print(message)


if not os.path.exists(REPORT_FOLDER):
    os.mkdir(REPORT_FOLDER)


# Checking if there is a gpu to speed learning up
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    log("CUDA is available")


# Function to open and read input file
def get_data_from_file():
    f = open(DATA_FILE, "r")
    return f.read()


# Function to create 10 sized vectors from the input
def create_digit_windows(digits, window_size):
    return_vectors = []
    for i in range(len(digits) - window_size + 1):
        vector = []
        for j in range(window_size):
            vector.append(digits[i + j])
        return_vectors.append(vector)
    return return_vectors


# Function to create the loaders from the vectors
def create_train_and_test_loaders_by_size(size):
    data = get_data_from_file()

    learn_data = data[:size]
    test_data = data

    learn_vectors = []
    test_vectors = []

    learn_labels = []
    test_labels = []

    learn_data = [int(i) for i in str(learn_data)]
    test_data = [int(i) for i in str(test_data)]

    learn_labels = np.array(learn_data[9:], dtype=float)
    test_labels = np.array(test_data[9:], dtype=float)

    learn_vectors = create_digit_windows(learn_data, 10)
    test_vectors = create_digit_windows(test_data, 10)

    learn_vectors = np.array(learn_vectors, dtype=float)
    test_vectors = np.array(test_vectors, dtype=float)

    # Creating datasets from the data
    train_set = TensorDataset(torch.from_numpy(learn_vectors), torch.from_numpy(learn_labels))
    test_set = TensorDataset(torch.from_numpy(test_vectors), torch.from_numpy(test_labels))

    train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=BATCH_SIZE)

    return train_loader, test_loader


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = 1
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        x = x.view(BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE)
        batch_size = numbers.shape[0]
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
    

        out, _ = self.gru(x, h0)

        out = out[:, -1, :]

        out = self.fc(out)
        return out


# result arrays to write to csv, with header row
with open(REPORT_FILE, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Hidden_dimension', 'Set_size', 'Accuracy'])
    for i in range(0, 9, 1):
        HIDDEN_SIZE = pow(2, i)

        for j in range(2, 7):
            set_sizes = pow(10, j)

            train_loader, test_loader = create_train_and_test_loaders_by_size(set_sizes)

            model = GRU(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(DEVICE)

            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

            n_total_steps = len(train_loader)
            loss_avg = 100
            number_epochs = 0
            while loss_avg > 1.7:
                loss_avg = 0
                number_epochs = number_epochs + 1
                for i, (numbers, labels) in enumerate(train_loader):

                    numbers = numbers.to(DEVICE)
                    labels = labels.to(DEVICE)

                    # Forward pass
                    outputs = model(numbers.float())
                    loss = criterion(outputs, labels.long())

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_avg = loss_avg + loss.item()

                loss_avg = loss_avg / n_total_steps
                print('loss: ' + str(loss_avg))

            with torch.no_grad():
                result = []
                n_correct = 0
                n_samples = 0
                for numbers, labels in test_loader:
                    numbers = numbers.to(DEVICE)
                    labels = labels.to(DEVICE)

                    outputs = model(numbers.float())
                    # max returns (value ,index)
                    _, predicted = torch.max(outputs.data, 1)
                    n_samples += labels.size(0)
                    n_correct += (predicted == labels).sum().item()

                acc = 100.0 * n_correct / n_samples
                result.append(str(HIDDEN_SIZE))
                result.append(str(set_sizes))
                result.append(str(acc))
                #results.append(result)
                writer.writerow(result)


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv("D:/Thesis 2/report/model_results.csv")

# print out Accuracy's relation with Set Size under same hidden dimension
grouped = results.groupby('Hidden_dimension')
for dim, group in grouped:
    filename = f'Result with {dim} hidden dimension'
    plt.plot(group['Set_size'].apply(str), group['Accuracy'])
    plt.xlabel('Set Size')
    plt.ylabel('Accuracy')
    plt.title(filename)
    plt.savefig("D:/Thesis 2/report/" + filename)
    plt.clf()

# Comparison
grouped = results.groupby('Hidden_dimension')
for dim, group in grouped:
    filename = 'Result comparison dimension as legend'
    plt.plot(group['Set_size'].apply(str), group['Accuracy'], label=f"Hidden Dim {dim}")
    plt.xlabel('Set Size')
    plt.ylabel('Accuracy')
    # todo: change title
    plt.title(filename)
    plt.legend()
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #       fancybox=True, shadow=True, ncol=5)
plt.savefig("D:/Thesis 2/report/" + filename)
plt.clf()

# print out Accuracy's relation with Hidden Dimension under same set size
grouped = results.groupby('Set_size')
for set_size, group in grouped:
    filename = f'Result with set size {set_size}'
    plt.plot(group['Hidden_dimension'].apply(str), group['Accuracy'])
    plt.xlabel('Hidden Dimension')
    plt.ylabel('Accuracy')
    plt.title(filename)
    plt.savefig("D:/Thesis 2/report/" + filename)
    plt.clf()

# Comparisons
grouped = results.groupby('Set_size')
for dim, group in grouped:
    filename = 'Result comparison set size as legend'
    plt.plot(group['Hidden_dimension'].apply(str), group['Accuracy'], label=f"Set Size {dim}")
    plt.xlabel('Hidden Dimension')
    plt.ylabel('Accuracy')
    # todo: change title
    plt.title(filename)
    plt.legend()
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #       fancybox=True, shadow=True, ncol=5)
plt.savefig("D:/Thesis 2/report/" + filename)
plt.clf()


# In[ ]:




