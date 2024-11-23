import re
import nltk
import string
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
from nltk.tokenize import word_tokenize

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)   # Remove numbers
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    #stemmer = PorterStemmer()
    #text = " ".join([stemmer.stem(word) for word in text.split()])
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text.split() if not word in set(STOPWORDS)]
    text = ' '.join(text)
    return text

def handle_negations(tokens):
    negated = []
    negation = False
    for word in tokens:
        if word == "not":
            negation = True
        elif negation:
            negated.append("not_" + word)
            negation = False
        else:
            negated.append(word)
    return negated
