import io
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
from nltk.tokenize import word_tokenize

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)


# Function to replace all instances of newlines or tabs with a single whitespace
def remove_newlines_tabs(text):
    formatted_text = text.replace('\\n', ' ').replace('\n', ' ').replace('\t',' ').replace('\\', ' ')
    return formatted_text


# Function to remove extra whitespaces
def remove_whitespace(text):
    pattern = re.compile(r'\s+')
    without_whitespace = re.sub(pattern, ' ', text)
    return without_whitespace


# Function to convert all characters to lowercase
def lower_casing(text):
    text = text.lower()
    return text


# Function to remove all special characters that aren't required. Special characters retained = . %
def remove_char(text):
    text = text.replace('(', '').replace(')', '').replace(',','')
    text = re.sub(r"[^0-9a-zA-Z%.]+", ' ', text)
    return text


# We will also remove stopwords as some samples contain them
stoplist = stopwords.words('english')
stoplist = set(stoplist)
def remove_stopwords(text):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stoplist]
    text = (" ").join(tokens_without_sw)
    return text


# Function to remove URLs from the text
def remove_urls(text):
    text = re.sub(r'http\S+', ' ', text)
    return text.strip()


# Function to remove emoticons, emojis or any pictorial representation
def remove_emojis(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r' ', text)


# Function to lemmatize our text
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    lemmatized_tokens = [lemma.lemmatize(token, pos = 'v') for token in tokens]
    return lemmatized_tokens

def lemmatize(text):
    text_tokens = word_tokenize(text)
    lemmatized_tokens = lemmatize_tokens(text_tokens)
    text = (" ").join(lemmatized_tokens)
    return text


# Combining all the above functions into a single data preprocessing function
def format_text(text):
    text = remove_newlines_tabs(text)
    text = remove_whitespace(text)
    text = remove_emojis(text)
    text = lower_casing(text)
    text = remove_char(text)
    text = remove_urls(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    return text


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

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_length):
        self.vocabulary = list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
        self.labels = [labels[label] for label in dataset['tag']]
        texts = []
        for string in dataset['text']:
            text = ""
            for tx in string:
                text += tx
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
        data = np.array([self.vocabulary.index(i) for i in list(raw_text) if i in self.vocabulary],
                        dtype=np.dtype(int))
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.concatenate(
                (data, np.zeros((self.max_length - len(data)), dtype=np.dtype(int))), axis=None)
        elif len(data) == 0:
            data = np.zeros((self.max_length), dtype=np.dtype(int))
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
    trainset, valset, testset = np.split(dataset.sample(frac=1, random_state=42),[int(.8*len(dataset)), int(.9*len(dataset))])
    
    # Creating character embeddings dataset from our text dataset
    trainset = Dataset(trainset, 64)
    valset = Dataset(valset, 64)
    testset = Dataset(testset, 64)
    
    return trainset, valset, testset


class Classifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_classes=11):
        super(Classifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.fc = nn.Linear(embedding_dim*64, num_classes)

    def forward(self, input):
        embedded = self.embedding(input)
        x = embedded.view(embedded.size(0), -1)  # Flatten
        output = self.fc(x)
        return output


# Function to load objects to GPU
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


def train_embedding_layer(epochs=15, lr=3e-5):
    trainset, valset, testset = create_dataset()
    
    # Passing the newly formed datasets into a dataloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    
    # Initializing the model
    num_classes = 11
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Classifier(68, 64, 11).to(device)
    
    # Define loss function, optimizer, and evaluation metrics
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    training_loss = []
    validation_loss = []

    for i in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.type(torch.LongTensor), labels.type(torch.LongTensor)
            inputs, labels = to_device(inputs, device), to_device(labels, device)
            predictions = model(inputs)
            loss = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            running_loss += loss.item()

            optimizer.step()

        # Choosing the class with maximum probability as the predicted class
        _, predicted = torch.max(predictions.data,1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Finding out training loss
        train_loss = running_loss/len(trainloader)
        training_loss.append(train_loss)

        # Finding out accuracy of predictions of that particular batch of training data
        accu = 100.*correct/total

        if ((i+1)%1==0):
            print('Epoch [%d / %d] Train Loss: %.3f | Accuracy: %.3f'%((i+1), epochs, train_loss, accu))

        # Finding out validation loss
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valloader:
                # inputs, labels = inputs.to(device), labels.to(device)
                inputs, labels = inputs.type(torch.LongTensor), labels.type(torch.LongTensor)
                inputs, labels = to_device(inputs, device), to_device(labels, device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

        val_loss = running_loss/len(valloader)
        validation_loss.append(val_loss)
        
        
    # Finding out the accuracy of the model on train and validation data
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in trainloader:
            inputs, labels = inputs.type(torch.LongTensor), labels.type(torch.LongTensor)
            inputs, labels = to_device(inputs, device), to_device(labels, device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'The training set accuracy is : {100 * correct // total} %')

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs,labels in valloader:
            inputs, labels = inputs.type(torch.LongTensor), labels.type(torch.LongTensor)
            inputs, labels = to_device(inputs, device), to_device(labels, device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'The validation set accuracy is : {100 * correct // total} %')
    
    
    # Finding out the accuracy on the testing data
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs,labels in testloader:
            inputs, labels = inputs.type(torch.LongTensor), labels.type(torch.LongTensor)
            inputs, labels = to_device(inputs, device), to_device(labels, device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'The testing set accuracy is : {100 * correct // total} %')
    
    
    # Plotting training loss and validation loss vs no. of epochs

    plt.figure(figsize=(8,4))
    plt.plot([(i+1) for i in range(epochs)], training_loss, 'b')
    plt.plot([(i+1) for i in range(epochs)], validation_loss, 'r')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['training loss', 'validation loss'])
    plt.title("Training the Embedding Layer")
    plt.savefig('embedding_layer_loss.png')
    
    
    torch.save(model.embedding.weight, 'embeddings.pt')


train_embedding_layer(epochs=50, lr=1e-5, batch_size=4)
