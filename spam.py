# Step 1: Import libraries
import pandas as pd
import nltk
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from nltk.corpus import stopwords

# Download stopwords (run once)
nltk.download('stopwords')

# Step 2: Load dataset
data = pd.read_csv("spam.csv", encoding="latin1")
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Step 3: Convert labels to numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Step 4: Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)        # remove symbols
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

data['message'] = data['message'].apply(clean_text)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# Step 6: TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 7: Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 8: Evaluate model
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 9: Custom prediction
def predict_spam(text):
    text = clean_text(text)
    text_tfidf = vectorizer.transform([text])
    result = model.predict(text_tfidf)
    return "SPAM ❌" if result[0] == 1 else "NOT SPAM ✅"

# Step 10: Test examples
emails = [
    "Congratulations! You won a free prize",
    "Congratulations! You won a free gift. Click this link http://bit.ly/free-offer",
    "Are we meeting tomorrow?",
    "URGENT! Your account has been blocked"
]

for e in emails:
    print(e, "→", predict_spam(e))