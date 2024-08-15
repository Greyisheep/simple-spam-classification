from src.train import main as train_main
from src.predict import predict_spam

if __name__ == "__main__":
    # Train the model (uncomment if needed)
    # train_main()
    
    # Predict on a new email
    new_email = """
    Hello, please visit http://example.com for more details!
    """
    print(predict_spam(new_email))
