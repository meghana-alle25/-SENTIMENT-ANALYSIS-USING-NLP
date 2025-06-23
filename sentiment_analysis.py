import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step 1: Load dataset
df = pd.read_csv("reviews.csv")

# Step 2: Preprocess
df.dropna(inplace=True)
X = df['review']
y = df['sentiment']

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Predict on new data
sample_reviews = ["Amazing product!", "I hate it."]
sample_tfidf = vectorizer.transform(sample_reviews)
predictions = model.predict(sample_tfidf)
for review, sentiment in zip(sample_reviews, predictions):
    print(f"Review: '{review}' => Sentiment: {'Positive' if sentiment == 1 else 'Negative'}")

# Step 8: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot()
plt.show()
