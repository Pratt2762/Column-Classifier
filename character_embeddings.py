import torch
import numpy as np
import pandas as pd
from data_preprocessing import format_text

labels = {'Employers_Payment':0,
          'EmployeesPayment':1,
          'Fund_Name':2,
          'Payments_Type_and_Date':3,
          'Higher_Rate':4,
          'Middle_rate':5,
          'Lower_Rate':6,
          'Number_of_Units':7,
          'Unit_price':8,
          'Fund_Value':9,
          'Split':10
          }


# Creating the dataset class to create character embeddings from the given dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_length=256):
        self.vocabulary = list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
        self.identity_mat = np.identity(len(self.vocabulary))
        self.labels = [labels[label] for label in dataset['tag']]
        texts = []
        for string in dataset['text']:
            text = ""
            for tx in string:
                text += tx
                text += " "
            texts.append(text)
        self.texts = texts
        self.max_length = max_length
        self.length = len(self.labels)
        self.num_classes = len(set(self.labels))

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, index):
        raw_text = self.texts[index]
        data = np.array([self.identity_mat[self.vocabulary.index(i)] for i in list(raw_text) if i in self.vocabulary],
                        dtype=np.float32)
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.concatenate(
                (data, np.zeros((self.max_length - len(data), len(self.vocabulary)), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros((self.max_length, len(self.vocabulary)), dtype=np.float32)
        label = self.labels[index]
        return data, label
      
      
def create_dataset():
    # Define the dataset
    dataset = pd.read_csv("data.csv")
    dataset.drop(columns = dataset.columns[0], inplace=True, axis=1)

    # Function to drop all NaN values
    dataset = dataset.dropna()

    # Dropping the rows which don't have a tag
    dataset = dataset[dataset.tag != '-']

    # Converting all items in the 'text' column to dtype string
    dataset['text'] = dataset['text'].map(str)
    
    # Applying all the necessary preprocessing techniques to our data
    for i in range(dataset.shape[0]):
        dataset.iloc[i,1] = format_text(dataset.iloc[i,1])
    
    # Splitting our data into train, val and test data
    np.random.seed(42)
    trainset, valset = np.split(dataset.sample(frac=1, random_state=42),[int(.8*len(dataset))])
    
    # Creating character embeddings dataset from our text dataset
    trainset = Dataset(trainset, 256)
    valset = Dataset(valset, 256)
    
    return trainset, valset
