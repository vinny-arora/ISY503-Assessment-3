

# Redefine the function to load reviews from a file as plain text, manually extracting review texts
def load_reviews_plain_text(file_path):
    reviews = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
            # Splitting the data into individual reviews based on a likely delimiter
            review_texts = data.split('<review_text>')
            for review in review_texts[1:]:  # Skipping the first split as it's likely not a review
                end_index = review.find('</review_text>')
                if end_index != -1:
                    text = review[:end_index].strip()
                    reviews.append(text)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    return reviews



# Updated function to load reviews from each domain
def load_domain_reviews(domain_path):
    import os
    positive_path = os.path.join(domain_path, 'positive.review')
    negative_path = os.path.join(domain_path, 'negative.review')

    positive_reviews = load_reviews_plain_text(positive_path) if os.path.exists(positive_path) else []
    negative_reviews = load_reviews_plain_text(negative_path) if os.path.exists(negative_path) else []

    return positive_reviews, negative_reviews



# Function to clean the text data
def clean_text(text):
    import string
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return ' '.join(words)





def main():
    import tarfile
    import os

    # Path to the uploaded tar.gz file
    tar_file_path = 'domain_sentiment_data.tar.gz'

    # Extract the tar.gz file
    extracted_folder_path = '/data/extracted_data'
    os.makedirs(extracted_folder_path, exist_ok=True)

    with tarfile.open(tar_file_path, 'r:gz') as tar:
        tar.extractall(path=extracted_folder_path)

    # List the contents of the extracted folder to understand the directory structure
    extracted_contents = os.listdir(extracted_folder_path)
    
    # List the contents of the 'sorted_data_acl' directory
    sorted_data_path = os.path.join(extracted_folder_path, 'sorted_data_acl')
    domains = os.listdir(sorted_data_path)
    domains_path = [os.path.join(sorted_data_path, domain) for domain in domains]

    # Display the directory structure for each domain
    domain_structure = {domain: os.listdir(domain_path) for domain, domain_path in zip(domains, domains_path)}
    # Re-run the process of loading the domain reviews
    all_positive_reviews = []
    all_negative_reviews = []
    
    for domain_path in domains_path:
        pos_reviews, neg_reviews = load_domain_reviews(domain_path)
        all_positive_reviews.extend(pos_reviews)
        all_negative_reviews.extend(neg_reviews)
    
    # Re-combine and re-label the data
    all_reviews = all_positive_reviews + all_negative_reviews
    all_labels = [1] * len(all_positive_reviews) + [0] * len(all_negative_reviews)  # 1 for positive, 0 for negative
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    import string
    # Clean all reviews
    cleaned_reviews = [clean_text(review) for review in all_reviews]

    # Outlier Removal: Remove very short reviews (less than, say, 10 words)
    min_words = 10
    filtered_reviews = [review for review in cleaned_reviews if len(review.split()) >= min_words]
    filtered_labels = [all_labels[i] for i, review in enumerate(cleaned_reviews) if len(review.split()) >= min_words]

    # Tokenization and Padding/Truncation of the reviews
    max_vocab_size = 10000  # Maximum number of unique words
    max_length = 100        # Maximum length of the reviews

    # Data Encoding using CountVectorizer
    vectorizer = CountVectorizer(max_features=max_vocab_size)
    vectorized_data = vectorizer.fit_transform(filtered_reviews).toarray()

    # Splitting the data into training, validation, and test sets (60%, 20%, 20% split)
    train_ratio = 0.6
    validation_ratio = 0.2
    test_ratio = 0.2

    # Train split
    X_train, X_temp, y_train, y_temp = train_test_split(vectorized_data, filtered_labels, test_size=(1 - train_ratio), random_state=42)

    # Validation and Test split
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_ratio/(test_ratio + validation_ratio), random_state=42)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score

    # Define and instantiate the Logistic Regression model
    logistic_model = LogisticRegression(max_iter=1000)

    # Train the model on the training data
    logistic_model.fit(X_train, y_train)

    # Test the model on the test set
    y_pred = logistic_model.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    print("Accuracy: ", accuracy)
    print("Classification Report:\n", classification_rep)
    import joblib

    # Save the model to a file
    model_filename = 'logistic_model.joblib'
    joblib.dump(logistic_model, model_filename)
    # Save the CountVectorizer vocabulary
    vocab_filename = 'vectorizer_vocab.joblib'
    joblib.dump(vectorizer.vocabulary_, vocab_filename)
    
    import joblib
    from sklearn.feature_extraction.text import CountVectorizer
    import string

    # Load the trained model and the vocabulary
    model = joblib.load('logistic_model.joblib')
    loaded_vocab = joblib.load('vectorizer_vocab.joblib')
    
    # User input
    user_input = input("Enter a review: ")

    # Clean and preprocess the input
    cleaned_input = clean_text(user_input)
    vectorizer = CountVectorizer(max_features=max_vocab_size, vocabulary=loaded_vocab)
    processed_input = vectorizer.transform([cleaned_input])

    # Predict sentiment
    prediction = model.predict(processed_input)[0]
    sentiment = "Positive review" if prediction == 1 else "Negative review"

    # Display the result
    print(sentiment)

if __name__ == "__main__":
    main()
