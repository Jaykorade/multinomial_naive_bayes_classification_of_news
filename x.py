import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
newsgroups = pd.read_csv('input.csv')

# Data and labels
X = newsgroups.short_description
y = newsgroups.category

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=10000)

# Fit and transform the training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform the test data
X_test_tfidf = tfidf.transform(X_test)
# Initialize Naive Bayes classifier
nb = MultinomialNB()

# Train the model
nb.fit(X_train_tfidf, y_train)
# Predict on test data
y_pred = nb.predict(X_test_tfidf)

# Print classification report
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
