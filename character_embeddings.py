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
    def __init__(self, dataset, max_length):
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
      
      
# Creating the character embeddings of each sample from our dataset
character_embeddings = Dataset(dataset, 1000)

# Passing the newly formed dataset into a dataloader
dataloader = torch.utils.data.DataLoader(character_embeddings, batch_size=2, shuffle=False)

# Checking to see whether our code works
for inputs, labels in dataloader:
    print("Input -")
    print(inputs)
    print("\nLabel -")
    print(labels)
    break
