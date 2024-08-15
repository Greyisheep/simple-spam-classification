import re
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

import warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

nltk.download('stopwords')
nltk.download('punkt')

def preprocess_email(email):
    """
    Preprocess a single email by cleaning, tokenizing, and removing stopwords.

    :param email: Raw email text or a file path.
    :return: Cleaned and tokenized email text.
    """
    # Check if input is a file path
    if re.match(r'^[\w,\s-]+\.[A-Za-z]{3}$', email):
        # If it's a file path, open the file and read the content
        with open(email, 'r') as f:
            email_text = f.read()
    else:
    # If it's not a file path, treat it as email text
        email_text = email

    # Remove HTML tags if present (assuming email_text now contains the content)
    soup = BeautifulSoup(email_text, "html.parser")
    email_text = soup.get_text()
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(email_text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stopwords.words('english')]
    
    # Replace URLs and email addresses with placeholders
    tokens = [re.sub(r'\bhttps?://\S+\b', 'url', t) for t in tokens]
    tokens = [re.sub(r'\b\S+@\S+\.\S+\b', 'email_address', t) for t in tokens]
    
    return ' '.join(tokens)

def preprocess_data(df, text_column):
    """
    Apply preprocessing to the entire DataFrame.
    
    :param df: DataFrame containing the email data.
    :param text_column: Name of the column containing email text.
    :return: DataFrame with processed email text.
    """
    df[text_column] = df[text_column].apply(preprocess_email)
    return df
