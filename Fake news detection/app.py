import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv(r"C:\Users\Lenovo\Desktop\Fake news detection\news.csv")

print(df.shape)
print(df.head())

labels = df.label

x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100, 2)}%')

conf_matrix = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
print("Confusion Matrix:")
print(conf_matrix)

random_indices = np.random.randint(0, len(df), 5)
random_news = df.iloc[random_indices]['text']
tfidf_random = tfidf_vectorizer.transform(random_news)
predicted_labels = pac.predict(tfidf_random)

for i, news in enumerate(random_news):
    print(f"News Text:\n{news[:200]}...\nPredicted Label: {predicted_labels[i]}\n")
