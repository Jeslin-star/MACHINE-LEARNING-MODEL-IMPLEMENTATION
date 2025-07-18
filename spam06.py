import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# === Step 1: Load Dataset ===
data = pd.read_csv("/storage/emulated/0/Download/spam_sample_laiyana.csv", encoding='utf-8', quotechar='"', on_bad_lines='skip')

# === Step 2: Preprocess Text ===
data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

data['cleaned_msg'] = data['message'].apply(clean_text)

# === Step 3: Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    data['cleaned_msg'], data['label_num'], test_size=0.3, random_state=42)

# === Step 4: Vectorization ===
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === Step 5: Train Model ===
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# === Step 6: Evaluate Model ===
y_pred = model.predict(X_test_vec)
print("=== Spam Detection Report ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === Step 7: Predict Smartly (Using Keywords + ML) ===
def predict_email(message):
    spam_keywords = ["lottery", "win", "jackpot", "prize", "click", "claim", "free", "urgent", "money", "offer"]
    
    cleaned = clean_text(message)
    if len(cleaned.split()) < 2:
        return "Not enough information to classify the message."
    
    # Check manually for spam keywords
    if any(word in cleaned for word in spam_keywords):
        return "SPAM Email Detected"
    
    return "Ham message. No spam detected in this"

# === Step 8: Input Loop ===
while True:
    print("\nEnter a message to check (or type 'exit'):")
    user_input = input("> ")
    if user_input.lower().strip() == "exit":
        print("Exiting program.")
        break
    print("Prediction:", predict_email(user_input))
