import unittest
from flask_app.app import app

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Sentiment Analysis</title>', response.data)

    def test_predict_page(self):
        response = self.client.post('/predict', data=dict(text="I love this!"))
        self.assertEqual(response.status_code, 200)

        # ✅ Check for all 13 possible sentiments
        expected_labels = [
            b'anger', b'boredom', b'empty', b'enthusiasm', b'fun',
            b'happiness', b'hate', b'love', b'neutral', b'relief',
            b'sadness', b'surprise', b'worry'
        ]

        self.assertTrue(
            any(label in response.data for label in expected_labels),
            "Response should contain one of the expected sentiment labels"
        )

if __name__ == '__main__':
    unittest.main()
