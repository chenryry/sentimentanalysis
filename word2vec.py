from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score, f1_score
import pandas as pd
import numpy as np


def get_text_vector(tokens, model):
    word_vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)
def word2vec(df):
    word2vec_model = Word2Vec(df['tokens'], vector_size=100, window=5, min_count=1, workers=4)
    X = np.array([get_text_vector(tokens, word2vec_model) for tokens in df['tokens']])
    Y = df["Sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=21, stratify = Y)
    model = RandomForestClassifier(class_weight='balanced', random_state = 21, n_estimators= 50, min_samples_split=5,min_samples_leaf=2)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return accuracy_score(y_pred,y_test),f1_score(y_pred,y_test, average='weighted')