import pandas as pd
from sklearn.model_selection import train_test_split
from data.load_data import load_data, save_processed_data
from src.preprocess import preprocess_data
from models.model import create_model, train_model, evaluate_model
from models.save_load_model import save_model

def main():
    # Load and preprocess data
    data = load_data('data/raw/emails.csv')
    data = preprocess_data(data, 'text')
    save_processed_data(data, 'data/processed/processed_emails.csv')

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['spam'], test_size=0.2, random_state=42)

    # Create and train model
    model = create_model()
    train_model(model, X_train, y_train)

    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    print(f'Model Accuracy: {accuracy}')

    # Save model
    save_model(model, 'models/spam_classifier.joblib')
    
if __name__ == "__main__":
    main()
