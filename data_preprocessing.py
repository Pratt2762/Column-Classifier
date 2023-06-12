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
  

# Combining all the above functions into a single data preprocessing function
def format_text(text):
  text = remove_newlines_tabs(text)
  text = remove_whitespace(text)
  text = lower_casing(text)
  text = remove_char(text)
  text = remove_stopwords(text)
  return text
