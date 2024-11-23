import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from util import preprocess,handle_negations
from tfidf import tfidf
from word2vec import word2vec, get_text_vector

df = pd.read_csv("data.csv")
df['Sentence'] = df['Sentence'].apply(preprocess)
df['tokens'] = df['Sentence'].apply(word_tokenize)
df['tokens'] = df['tokens'].apply(handle_negations)

print(tfidf(df))
print(word2vec(df))
