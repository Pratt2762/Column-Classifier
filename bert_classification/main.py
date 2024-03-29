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
from synthetic_data_addition import add_synthetic_data

from bert_classifier import BertClassifier, train, evaluate_bert, evaluate_classwise_bert


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


# # Adding some extra synthetic data (manually generated) to our data to increase its diversity
# dataset = add_synthetic_data(dataset)


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

# Function to determine the size of our model in order to keep an eye on computational overhead
def find_model_size(model):
  param_size = 0
  for param in model.parameters():
      param_size += param.nelement() * param.element_size()
  buffer_size = 0
  for buffer in model.buffers():
      buffer_size += buffer.nelement() * buffer.element_size()

  size_all_mb = (param_size + buffer_size) / 1024**2
  print('model size: {:.3f}MB'.format(size_all_mb))

find_model_size(model)

# Evaluating its performance 
evaluate_bert(model, testset)
evaluate_classwise_bert(model, testset)
