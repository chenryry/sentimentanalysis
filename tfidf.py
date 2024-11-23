from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import pandas as pd
import numpy as np

def tfidf(df):
    df['joined_text'] = df['tokens'].apply(lambda x: ' '.join(x))


    tfidf_vectorizer = TfidfVectorizer(max_df=0.7, min_df=2)
    X = tfidf_vectorizer.fit_transform(df['joined_text'])

    Y = df["Sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.25, random_state=21)
    model = RandomForestClassifier(class_weight='balanced', random_state = 21, n_estimators= 50, min_samples_split=5,min_samples_leaf=2)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return accuracy_score(y_pred,y_test),f1_score(y_pred,y_test, average='weighted')
