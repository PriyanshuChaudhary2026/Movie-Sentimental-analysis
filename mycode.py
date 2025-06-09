# -------------------------------
# Import required libraries
# -------------------------------
import pandas as pd  # For loading and manipulating data
from sklearn.model_selection import train_test_split, GridSearchCV  # For splitting and hyperparameter tuning
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text to TF-IDF features
from sklearn.pipeline import Pipeline  # For chaining TF-IDF and model steps
from sklearn.svm import LinearSVC  # Support Vector Machine model for classification
from sklearn.metrics import accuracy_score, classification_report  # For evaluating model performance

# -------------------------------
# Load and preprocess the IMDB dataset
# -------------------------------
sentiment_df = pd.read_csv(
    'IMDB Dataset.csv',
    encoding='utf-8',         # Handle special characters properly
    on_bad_lines='skip',      # Skip any malformed lines in the CSV
    engine='python'           # Use Python engine to tolerate irregular CSV formatting
)

# Keep only the first two columns if more exist (we expect: review and sentiment)
if sentiment_df.shape[1] > 2:
    sentiment_df = sentiment_df.iloc[:, :2]

# Rename the columns for clarity
sentiment_df.columns = ['review', 'sentiment']

# Convert sentiment values to binary: 'positive' → 1, 'negative' → 0
sentiment_df['sentiment'] = sentiment_df['sentiment'].map({'positive': 1, 'negative': 0})

# -------------------------------
# Split dataset into training and testing sets
# -------------------------------
X = sentiment_df['review']       # Input feature: review text
y = sentiment_df['sentiment']    # Output label: sentiment (1 or 0)

# 80% training, 20% testing split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Create a pipeline with TF-IDF and SVM
# -------------------------------
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),  # Convert text to TF-IDF vectors
    ('svc', LinearSVC())                                           # Linear Support Vector Classifier
])

# -------------------------------
# Define hyperparameter grid for GridSearchCV
# -------------------------------
param_grid = {
    'tfidf__max_df': [0.5, 0.7, 0.9],                      # Ignore very common words
    'tfidf__ngram_range': [(1, 1), (1, 2)],                # Try unigrams and bigrams
    'svc__C': [0.1, 1, 10]                                 # Regularization parameter for SVM
}

# -------------------------------
# Perform grid search with 5-fold cross-validation
# -------------------------------
grid_search = GridSearchCV(
    pipeline,           # Pipeline with TF-IDF + SVM
    param_grid,         # Hyperparameters to try
    cv=5,               # 5-fold cross-validation
    scoring='accuracy', # Use accuracy as evaluation metric
    n_jobs=-1,          # Use all CPU cores for faster training
    verbose=2           # Print detailed progress for each fold
)

# Fit the model on training data using grid search
grid_search.fit(X_train, y_train)

# -------------------------------
# Evaluate the best found model
# -------------------------------
best_model = grid_search.best_estimator_      # Get the best model from grid search
y_pred = best_model.predict(X_test)           # Predict sentiment on test data

# Print best hyperparameters and evaluation metrics
print("Best Parameters:", grid_search.best_params_)
print("SVM Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -------------------------------
# Function to predict sentiment of a single review
# -------------------------------
def predict_sentiment(review):
    prediction = best_model.predict([review])  # Predict using trained model
    return {0: "negative", 1: "positive"}.get(prediction[0], "unknown")  # Map result to label

# -------------------------------
# Take user input and predict sentiment
# -------------------------------
user_input = input("Enter a movie review: ")
print("Predicted Sentiment:", predict_sentiment(user_input))
