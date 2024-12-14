import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = AutoModel.from_pretrained("distilbert-base-cased")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


df = pd.read_csv("data.csv")
mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
df['Sentiment'] = df['Sentiment'].map(mapping)

texts = df["Sentence"].tolist()
labels = df["Sentiment"].tolist()


inputs = tokenizer(texts,padding=True,truncation=True,max_length=64,return_tensors="pt")
inputs = {key: val.to(device) for key, val in inputs.items()}

# Get [CLS] embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Extract [CLS] token embeddings for each sentence
cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Use the [CLS]-equivalent token
cls_embeddings = cls_embeddings.cpu().numpy()  # Move to CPU and convert to numpy

X_train, X_test, y_train, y_test = train_test_split(cls_embeddings, labels, test_size=0.3, stratify=labels, random_state=21)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))