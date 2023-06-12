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


# Define the dataset
dataset = pd.read_csv("data.csv")
dataset.drop(columns = dataset.columns[0], inplace=True, axis=1)

# Function to drop all NaN values
dataset = dataset.dropna()

# Converting all items in the 'text' column to dtype string
dataset['text'] = dataset['text'].map(str)

# Check to see if indeed the 'text' column is of dtype string
dataset.dtypes

# Applying the necessary preprocessing techniques to our data
for i in range(dataset.shape[0]):
  dataset.iloc[i,1] = format_text(dataset.iloc[i,1])
  
# Displaying our dataset after applying the preprocessing techniques to it
dataset

