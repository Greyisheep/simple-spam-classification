from models.save_load_model import load_model
from src.preprocess import preprocess_email

def predict_spam(email):
    """
    Predict whether an email is spam or not.
    
    :param email: Raw email text.
    :return: Prediction (spam or not spam).
    """
    model = load_model('models/spam_classifier.joblib')
    processed_email = preprocess_email(email)
    prediction = model.predict([processed_email])
    return 'spam' if prediction == 1 else 'not spam'
