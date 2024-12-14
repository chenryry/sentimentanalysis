Performed a Sentiment analysis using three different models on the same dataset of financial statements (tf-idf, word2vec, and BERT).  
The sentiment analysis dataset was created with two columns, one with the sentence of the financial statement called Sentence and the other column listing the sentiment named "Sentiment".  
In order to classify the data, RandomForestClassifier and Logistic Regression were used after using training the models above.  
In order to use the repo, download the folder and call main.py, which will run all the models and then return the accuracy of the model.  

Results of the Models:  
TF-IDF Random Forest: 0.513  
Word2Vec Random Forest: 0.604  
BERT Logistic Regression: 0.717  
