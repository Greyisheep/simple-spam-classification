import joblib

def save_model(model, file_path):
    """
    Save the trained model to a file.
    
    :param model: Trained model pipeline.
    :param file_path: Path to save the model.
    """
    joblib.dump(model, file_path)

def load_model(file_path):
    """
    Load a trained model from a file.
    
    :param file_path: Path to the model file.
    :return: Loaded model pipeline.
    """
    return joblib.load(file_path)
