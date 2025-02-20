import requests
from textblob import TextBlob
import json
from datetime import datetime, timedelta
from config import *

class SentimentAnalyzer:
    def __init__(self):
        # CryptoCompare API anahtarı
        self.crypto_compare_key = CRYPTOCOMPARE_API_KEY
    
    def analyze_news(self, symbol):
        """
        Haberleri analiz eder
        """
        try:
            url = f"https://min-api.cryptocompare.com/data/v2/news/?categories={symbol}&api_key={self.crypto_compare_key}"
            response = requests.get(url)
            news = response.json()
            
            # Liste yerine string kullanın
            symbol_base = symbol.replace('USDT', '')
            
            sentiments = []
            for article in news['Data'][:10]:  # Son 10 haber
                analysis = TextBlob(article['title'] + " " + article['body'])
                sentiments.append(analysis.sentiment.polarity)
            
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)
                sentiment_score = self.normalize_sentiment(avg_sentiment)
                return sentiment_score
            return 0
            
        except Exception as e:
            print(f"Haber analizi hatası: {str(e)}")
            return 0
    
    def normalize_sentiment(self, sentiment_value):
        """
        Duygu skorunu -100 ile 100 arasına normalize eder
        """
        return sentiment_value * 100
    
    def get_overall_sentiment(self, symbol):
        """
        Haber analizlerini değerlendirir
        """
        news_sentiment = self.analyze_news(symbol)
        
        sentiment_status = "Nötr"
        if news_sentiment > 30:
            sentiment_status = "Çok Pozitif"
        elif news_sentiment > 10:
            sentiment_status = "Pozitif"
        elif news_sentiment < -30:
            sentiment_status = "Çok Negatif"
        elif news_sentiment < -10:
            sentiment_status = "Negatif"
        
        return {
            'skor': news_sentiment,
            'durum': sentiment_status,
            'haber_skor': news_sentiment
        } 