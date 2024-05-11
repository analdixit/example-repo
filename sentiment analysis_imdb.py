import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy

nltk.download('movie_reviews')

def extract_features(words):
    return dict([(word, True) for word in words])

# Get movie reviews with their categories (positive/negative)
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents
import random
random.shuffle(documents)

# Extract features from the reviews
featuresets = [(extract_features(words), category) for (words, category) in documents]

# Split the featuresets into training and testing datasets
train_set, test_set = featuresets[:1600], featuresets[1600:]

classifier = NaiveBayesClassifier.train(train_set)

print("Accuracy:", nltk_accuracy(classifier, test_set))

# Example review
review = "This movie is fantastic! I loved it."

# Tokenize the review
words = nltk.word_tokenize(review)

# Extract features from the review
features = extract_features(words)

# Perform sentiment analysis
print("Sentiment:", classifier.classify(features))
