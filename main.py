import io
import re

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.tokenize import word_tokenize

import data_preprocessing
from data_preprocessing import format_text

from bert_classifier import BertClassifier, train, evaluate


# Define the dataset
dataset = pd.read_csv("data.csv")
dataset.drop(columns = dataset.columns[0], inplace=True, axis=1)

# Function to drop all NaN values
dataset = dataset.dropna()

# Dropping the rows which don't have a tag
dataset = dataset[dataset.tag != '-']

# Converting all items in the 'text' column to dtype string
dataset['text'] = dataset['text'].map(str)

# Check to see if indeed the 'text' column is of dtype string
dataset.dtypes

# Applying the necessary preprocessing techniques to our data
for i in range(dataset.shape[0]):
  dataset.iloc[i,1] = format_text(dataset.iloc[i,1])
  
# Displaying our dataset after applying the preprocessing techniques to it
dataset

# Displaying the distribution of our labels on a bar plot
dataset.groupby(['tag']).size().plot.bar()

# Finding out the number of instances of each label
dataset['tag'].value_counts()


# Splitting our dataset into train, val and test sets in the ratio 80:10:10
np.random.seed(42)
trainset, valset, testset = np.split(dataset.sample(frac=1, random_state=42), 
                                     [int(.8*len(dataset)), int(.9*len(dataset))])

print(len(trainset),len(valset), len(testset))


# Setting the training hyperparameters
EPOCHS = 5
model = BertClassifier()
LR = 1e-6

# Training the classifier on our data
train(model, trainset, valset, LR, EPOCHS)

# Evaluating its performance 
evaluate(model, testset)
