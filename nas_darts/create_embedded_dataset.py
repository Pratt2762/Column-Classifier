import numpy as np
import matplotlib.pyplot as plt


def get_embeddings():
    embeddings = torch.load('embeddings.pt')
    embeddings.requires_grad = False
    embeddings = embeddings.cpu()
    embeddings = embeddings.numpy()
    return embeddings


def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
            
    embeddings = get_embeddings()
    embeddings = embeddings + P
    return embeddings


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


class Embedded_Dataset(torch.utils.data.Dataset):
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

        embeddings = getPositionEncoding(seq_len=68, d=64, n=100)
        
        embedded_data = [embeddings[i] for i in data]
        embedded_data = np.array(embedded_data)
        
        return embedded_data, label


def create_embedded_dataset():
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
    trainset = Embedded_Dataset(trainset, 64)
    valset = Embedded_Dataset(valset, 64)
    testset = Embedded_Dataset(testset, 64)
    
    return trainset, valset, testset






