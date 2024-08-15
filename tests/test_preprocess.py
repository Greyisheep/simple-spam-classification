import unittest
from src.preprocess import preprocess_email

class TestPreprocessing(unittest.TestCase):
    
    def test_preprocess_email(self):
        raw_text = "<html>Hello, this is a <b>test</b> email!</html>"
        processed_text = preprocess_email(raw_text)
        self.assertNotIn('<b>', processed_text)
        self.assertIn('hello', processed_text)
        self.assertIn('test', processed_text)

if __name__ == "__main__":
    unittest.main()
