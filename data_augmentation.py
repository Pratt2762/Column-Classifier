# Applying augmentation techniques to our dataset

import nlpaug.augmenter.word as naw
aug = naw.SynonymAug(aug_src='wordnet',aug_max=2)    # a maximum of two words are going to be replaced with their synonyms

label_to_string = {'Employers_Payment':'employers payment',
                 'EmployeesPayment':'employees payment',
                 'Fund_Name':'fund name',
                 'Payments_Type_and_Date':'payments type and date',
                 'Higher_Rate':'higher rate',
                 'Middle_rate':'middle rate',
                 'Lower_Rate':'lower rate',
                 'Interest_rate':'interest rate',
                 'Income_yearly':'income yearly',
                 'Customer_satisfaction':'customer satisfaction',
                 'Number_of_Units':'number of units',
                 'Unit_price':'unit price',
                 'Fund_Value':'fund value',
                 'Split':'Split'
                }

augmented_dataset = {'tag':[], 'text':[]}


# Function to create new labels by reversing order of words in the labels
def reverse_label(label):
    string = label_to_string[label]
    words = string.split()
    reversed_label = '_'.join(reversed(words))
    return reversed_label


# Creating a function to augment the labels by replacing the reversed labels with synonyms (max 2)  
def augment_label(label):
    reversed_label = reverse_label(label)
    augmented_label = aug.augment(reversed_label, n=1)
    return augmented_label[0]


# Creating dicts for reversing and augmenting labels
reversed_tags = {}
augmented_tags = {}

for i in dataset['tag'].unique():
    reversed_tags[i] = reverse_label(i)
    augmented_tags[i] = augment_label(i)


# Creating a function to augment the texts by replacing the texts with synonyms (max 2)  
def augment_text(text):
    augmented_text = aug.augment(text, n=1)
    return augmented_text[0]
    

def augment_dataset(dataset):
    for i in range(len(dataset)):
        label = dataset.iloc[i,0]
        text = dataset.iloc[i,1]

        reversed_label = reversed_tags[label]
        augmented_label = augmented_tags[label]
        augmented_text = augment_text(text)

        # Adding a new entry with reversed label and augmented (synonym replacement) text
        augmented_dataset['tag'].append(reversed_label)
        augmented_dataset['text'].append(augmented_text)

        # Adding a new entry with augmented (reversed and synonym replacement) label and augmented text
        augmented_dataset['tag'].append(augmented_label)
        augmented_dataset['text'].append(augmented_text)


    augmented_dataset = pd.DataFrame(augmented_dataset)
    dataset = pd.concat([dataset, augmented_dataset])
    
    return dataset
