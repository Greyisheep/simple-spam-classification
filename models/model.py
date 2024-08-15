from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def create_model():
    """
    Create and return a spam detection model pipeline.
    
    :return: Model pipeline.
    """
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB()),
    ])
    return model

def train_model(model, X_train, y_train):
    """
    Train the model with training data.
    
    :param model: Model pipeline to be trained.
    :param X_train: Features for training.
    :param y_train: Labels for training.
    """
    model.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data.
    
    :param model: Trained model pipeline.
    :param X_test: Features for testing.
    :param y_test: Labels for testing.
    :return: Model's accuracy score.
    """
    return model.score(X_test, y_test)
