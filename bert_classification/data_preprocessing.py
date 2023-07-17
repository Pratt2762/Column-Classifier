import io
import re

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.tokenize import word_tokenize


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
