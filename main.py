import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from util import preprocess,handle_negations
from tfidf import tfidf
from word2vec import word2vec, get_text_vector
import mlflow
import mlflow.sklearn

df = pd.read_csv("data.csv")
df['Sentence'] = df['Sentence'].apply(preprocess)
df['tokens'] = df['Sentence'].apply(word_tokenize)
df['tokens'] = df['tokens'].apply(handle_negations)


mlflow.set_tracking_uri("http://127.0.0.1:5000")
with mlflow.start_run(run_name="TFIDF Random Forest"):
    accuracy_rf = tfidf(df)[0]
    f1_score = tfidf(df)[1]
    # Log parameters, metrics, and model
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", accuracy_rf)
    mlflow.sklearn.log_model(tfidf, "tfidf")

with mlflow.start_run(run_name="Word2Vec Random Forest"):
    accuracy_rf = word2vec(df)[0]
    f1_score = word2vec(df)[1]
    # Log parameters, metrics, and model
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", accuracy_rf)
    mlflow.sklearn.log_model(word2vec, "word2vec")

