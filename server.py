import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')
for review in reviews:
    sentiment = sia.polarity_scores(review['ReviewBody'])
    review['neg'] = sentiment['neg']
    review['neu'] = sentiment['neu']
    review['pos'] = sentiment['pos']
    review['compound'] = sentiment['compound']
    review['sentiment'] = sentiment

valid_locations = [
    "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California", "Colorado Springs, Colorado",
    "Denver, Colorado", "El Cajon, California", "El Paso, Texas", "Escondido, California", "Fresno, California",
    "La Mesa, California", "Las Vegas, Nevada", "Los Angeles, California", "Oceanside, California", 
    "Phoenix, Arizona", "Sacramento, California", "Salt Lake City, Utah", "San Diego, California", 
    "Tucson, Arizona"
]


class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body: str) -> dict:
        """
        Analyze sentiment of the review body text.
        """
        return sia.polarity_scores(review_body)

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        method = environ["REQUEST_METHOD"]
        if method == "GET":
            query_params = parse_qs(environ.get('QUERY_STRING', ''))
            location = query_params.get('location', [None])[0]
            start_date = query_params.get('start_date', [None])[0]
            end_date = query_params.get('end_date', [None])[0]

            filtered_reviews = reviews
            if location:
                filtered_reviews = [review for review in filtered_reviews if review['Location'] == location]
            if start_date:
                filtered_reviews = [review for review in filtered_reviews if review['Timestamp'] >= start_date]
            if end_date:
                filtered_reviews = [review for review in filtered_reviews if review['Timestamp'] <= end_date]

            filtered_reviews.sort(key=lambda x: x['compound'], reverse=True)

            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")
            
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

        elif method == "POST":
            try:
                request_size = int(environ.get('CONTENT_LENGTH', 0))
            except ValueError:
                request_size = 0

            request_body = environ['wsgi.input'].read(request_size).decode('utf-8')
            data = parse_qs(request_body)

            new_review_body = data.get('ReviewBody', [None])[0]
            new_location = data.get('Location', [None])[0]

            if not new_review_body or not new_location:
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                return [json.dumps({"error": "ReviewBody and Location are required"}).encode("utf-8")]

            if new_location not in valid_locations:
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                return [json.dumps({"error": "Invalid location"}).encode("utf-8")]

            new_id = str(uuid.uuid4())
            new_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            sentiment = self.analyze_sentiment(new_review_body)

            new_review = {
                "ReviewId": new_id,
                "ReviewBody": new_review_body,
                "Location": new_location,
                "Timestamp": new_timestamp,
                "sentiment": sentiment,
                "neg": sentiment['neg'],
                "neu": sentiment['neu'],
                "pos": sentiment['pos'],
                "compound": sentiment['compound']
            }

            reviews.append(new_review)

            response_body = json.dumps(new_review).encode("utf-8")

            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])

            return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8000))
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()