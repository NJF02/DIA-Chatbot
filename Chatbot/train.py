import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils import tokenise_lemmatise, bag_of_words
from model_framework import NeuralNet
from settings import train_path, data_path, sub_categories

class ChatDataset(Dataset):
    def __init__(self, x_train, y_train, transform = None):
        self.x = x_train
        self.y = y_train
        self.len = len(x_train)
        self.transform = transform

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

def preprocess_data(json_file):
    with open(json_file, 'r', encoding = 'utf-8') as f:
        intents = json.load(f)
        
    tags = []
    all_words = []
    xy = []
    # loop through each sentence in our intents patterns
    for intent in intents['intents']:
        tag = intent['tag']
        # add to tag list
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenise_lemmatise(pattern)
            all_words.extend(w)
            # add to xy pair
            xy.append((w, tag))

    # remove duplicates and sort
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    # display tags and words
    print(len(tags), "tags:", tags)
    print(len(all_words), "unique stemmed words:", all_words)
    print(len(xy), "patterns")
    
    return tags, all_words, xy

def train_model(tags, all_words, xy, data_file):
    # create training data
    x_train = []
    y_train = []
    for sentence, tag in xy:
        # bag of words for each pattern sentence
        bag = bag_of_words(sentence, all_words)
        x_train.append(bag)
        # class labels
        label = tags.index(tag)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    # hyper-parameters 
    num_epochs = 1000
    batch_size = 32
    learning_rate = 0.001
    input_size = len(all_words)
    output_size = len(tags)
    hidden_size = round((input_size + output_size) / 2)
    print(input_size, hidden_size, output_size)

    dataset = ChatDataset(x_train = x_train, y_train = y_train)
    train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 0)

    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # loss and optimiser
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr = learning_rate)

    # train the model
    for epoch in range(num_epochs):
        for words, labels in train_loader:
            words = words.to(device)
            labels = labels.to(dtype = torch.long).to(device)
            
            # feedforward
            outputs = model(words)
            loss = criterion(outputs, labels)
            
            # backpropagation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
        # print(epoch)
        if (epoch + 1) % (num_epochs / 10) == 0:
            print(f'Epoch [{epoch + 1} / {num_epochs}], Loss: {loss.item():.4f}')

    print(f'Final loss: {loss.item():.4f}')

    data = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'output_size': output_size,
        'tags': tags,
        'all_words': all_words,
        'model_state': model.state_dict()
    }

    torch.save(data, data_file)
    print(f'Training complete. File saved to {data_file}')
    
# train model for master file
json_file = train_path + "master.json"
tags, all_words, xy = preprocess_data(json_file)
data_file = data_path + "master.pth"
train_model(tags, all_words, xy, data_file)

for sub_category in sub_categories:
    json_file = train_path + sub_category + ".json"
    tags, all_words, xy = preprocess_data(json_file)
    data_file = data_path + sub_category + ".pth"
    train_model(tags, all_words, xy, data_file)