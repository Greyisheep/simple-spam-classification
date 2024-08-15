import pandas as pd

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    
    :param file_path: Path to the CSV file.
    :return: DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path)

def save_processed_data(df, file_path):
    """
    Save the processed DataFrame to a CSV file.
    
    :param df: DataFrame containing the processed data.
    :param file_path: Path where the processed CSV will be saved.
    """
    df.to_csv(file_path, index=False)
