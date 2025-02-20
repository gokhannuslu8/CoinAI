import ccxt
import numpy as np
from datetime import datetime, timedelta
import time
import pandas as pd
import json
import os
from sentiment_analyzer import SentimentAnalyzer
from data_collector import DataCollector
from sklearn.preprocessing import MinMaxScaler

class TradingBot:
    def __init__(self, model):
        self.model = model
        self.exchange = ccxt.binance()
        self.positions = {}  # Açık pozisyonları takip etmek için
        self.trade_history = []
        self.risk_per_trade = 0.02  # Hesap bakiyesinin %2'si
        self.max_open_positions = 3
        self.stop_loss_percent = 0.02  # %2
        self.take_profit_percent = 0.04  # %4
        
        # Trading sonuçlarını kaydetmek için klasör oluştur
        os.makedirs('trading_results', exist_ok=True)
        
        # Mevcut init koduna ekle:
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Data collector'ı başlat
        self.data_collector = DataCollector()
    
    def calculate_position_size(self, price, stop_loss_price):
        """
        Risk yönetimi: Pozisyon büyüklüğünü hesaplar
        """
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = balance['USDT']['free']
            risk_amount = usdt_balance * self.risk_per_trade
            position_size = risk_amount / (price - stop_loss_price)
            return position_size
        except Exception as e:
            print(f"Pozisyon büyüklüğü hesaplama hatası: {e}")
            return 0
            
    def calculate_levels(self, entry_price):
        """
        Stop-loss ve take-profit seviyelerini hesaplar
        """
        stop_loss = entry_price * (1 - self.stop_loss_percent)
        take_profit = entry_price * (1 + self.take_profit_percent)
        return stop_loss, take_profit
        
    def save_trade_history(self):
        """
        Trade geçmişini kaydeder
        """
        filename = f"trading_results/trades_{datetime.now().strftime('%Y%m%d')}.json"
        with open(filename, 'w') as f:
            json.dump(self.trade_history, f, indent=4, default=str)
            
    def analyze_volume(self, df):
        """
        Hacim analizi
        """
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
        
        if current_volume > avg_volume * 2:
            return 20  # Yüksek hacim
        elif current_volume < avg_volume * 0.5:
            return -20  # Düşük hacim
        return 0
        
    def analyze_trend(self, df):
        """
        Trend analizi
        """
        ma20 = df['close'].rolling(window=20).mean().iloc[-1]
        ma50 = df['close'].rolling(window=50).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        if current_price > ma20 and ma20 > ma50:
            return 15  # Yükselen trend
        elif current_price < ma20 and ma20 < ma50:
            return -15  # Düşen trend
        return 0
        
    def analyze_volatility(self, df):
        """
        Volatilite analizi
        """
        current_volatility = df['volatility'].iloc[-1]
        avg_volatility = df['volatility'].rolling(window=20).mean().iloc[-1]
        
        if current_volatility > avg_volatility * 2:
            return -20  # Çok yüksek volatilite - riskli
        elif current_volatility > avg_volatility * 1.5:
            return -10  # Yüksek volatilite
        elif current_volatility < avg_volatility * 0.5:
            return 10  # Düşük volatilite - daha güvenli
        return 0
    
    def analyze_signals(self, data, timeframe):
        """
        Genişletilmiş teknik analiz
        """
        try:
            df = data.get(timeframe)
            if df is None or len(df.index) == 0:
                return 0, {}
            
            signals = {
                'RSI': self.analyze_rsi(df) * 2.0,        # 1.5'ten 2.0'ye
                'MACD': self.analyze_macd(df) * 1.8,      # 1.2'den 1.8'e
                'BB': self.analyze_bollinger(df) * 1.5,   # Yeni ağırlık
                'StochRSI': self.analyze_stoch_rsi(df) * 1.8,  # 1.3'ten 1.8'e
                'Volume': self.analyze_volume(df) * 1.5,   # Yeni ağırlık
                'Trend': self.analyze_trend(df) * 2.0,     # 1.4'ten 2.0'ye
                'Volatility': self.analyze_volatility(df) * 1.5,  # Yeni ağırlık
                'ADX': self.analyze_adx(df) * 1.5          # Yeni ağırlık
            }
            
            total_score = sum(signals.values())
            return total_score, signals
        
        except Exception as e:
            print(f"Sinyal analizi hatası ({timeframe}): {e}")
            return 0, {}
    
    def analyze_rsi(self, df):
        """
        RSI analizi
        """
        rsi = df['RSI'].iloc[-1]
        if rsi < 30:
            return 30  # Aşırı satım
        elif rsi > 70:
            return -30  # Aşırı alım
        return 0
    
    def analyze_macd(self, df):
        """
        MACD analizi
        """
        macd = df['MACD'].iloc[-1]
        signal = df['MACD_Signal'].iloc[-1]
        
        if macd > signal:
            return 25  # Alış sinyali
        elif macd < signal:
            return -25  # Satış sinyali
        return 0
    
    def analyze_bollinger(self, df):
        """
        Bollinger Bands analizi
        """
        price = df['close'].iloc[-1]
        upper = df['BB_upper'].iloc[-1]
        lower = df['BB_lower'].iloc[-1]
        
        if price < lower:
            return 20  # Aşırı satım
        elif price > upper:
            return -20  # Aşırı alım
        return 0
    
    def analyze_stoch_rsi(self, df):
        """
        Stochastic RSI analizi
        """
        k = df['StochRSI_K'].iloc[-1]
        d = df['StochRSI_D'].iloc[-1]
        
        if k < 20 and d < 20:
            return 25  # Aşırı satım
        elif k > 80 and d > 80:
            return -25  # Aşırı alım
        return 0
    
    def analyze_adx(self, df):
        """
        ADX analizi (Trend gücü)
        """
        try:
            adx = df['ADX'].iloc[-1]
            dmp = df['DMP'].iloc[-1]  # Positive Directional Movement
            dmn = df['DMN'].iloc[-1]  # Negative Directional Movement
            
            if adx > 25:  # Güçlü trend
                if dmp > dmn:
                    return 15  # Güçlü yükseliş trendi
                elif dmn > dmp:
                    return -15  # Güçlü düşüş trendi
            return 0
        except Exception as e:
            print(f"ADX analiz hatası: {str(e)}")
            return 0
    
    def get_trading_decision(self, symbol, data):
        """
        Trading kararı verir
        """
        if data is None or not isinstance(data, dict) or not data:
            return {
                'final': {
                    'signal': 'BEKLE',
                    'güven': 0,
                    'trend_yönü': 'Veri Yok'
                }
            }
        
        # Duygu analizini al
        sentiment = self.sentiment_analyzer.get_overall_sentiment(symbol)
        
        decisions = {}
        weights = {
            '4h': 0.4,    # Ana trend için yüksek ağırlık
            '1h': 0.4,    # Orta vadeli sinyaller için yüksek ağırlık
            '15m': 0.2    # Kısa vadeli teyit
        }
        
        final_decision = {
            'signal': 'BEKLE',
            'güven': 0,
            'trend_yönü': 'Belirsiz'
        }
        
        total_score = 0
        valid_timeframes = 0
        
        for timeframe, df in data.items():
            if df is None or len(df.index) == 0:
                continue
            
            valid_timeframes += 1
            score, indicators = self.analyze_signals(data, timeframe)
            prediction = self.model.predict(self.prepare_data_for_prediction(df))
            
            timeframe_score = (
                score * 0.5 +
                float(prediction[0][0]) * 0.3 +
                sentiment['skor'] * 0.2
            )
            
            weighted_score = timeframe_score * weights.get(timeframe, 0.2)
            total_score += weighted_score
            
            decisions[timeframe] = {
                'score': float(timeframe_score),  # numpy.float64'ü normal float'a çevir
                'weighted_score': float(weighted_score),
                'signal': 'AL' if timeframe_score > 50 else 'SAT' if timeframe_score < -50 else 'BEKLE',
                'güven': float(abs(timeframe_score)),
                'indicators': indicators,
                'sentiment': sentiment
            }
        
        if valid_timeframes == 0:
            return {
                'final': {
                    'signal': 'BEKLE',
                    'güven': 0,
                    'trend_yönü': 'Yetersiz Veri'
                }
            }
        
        # Final kararı belirle
        if total_score > 15:
            final_decision['signal'] = 'AL'
            final_decision['trend_yönü'] = 'Yükseliş'
        elif total_score < -15:
            final_decision['signal'] = 'SAT'
            final_decision['trend_yönü'] = 'Düşüş'
        
        final_decision['güven'] = float(abs(total_score))  # numpy.float64'ü normal float'a çevir
        decisions['final'] = final_decision
        
        return decisions
    
    def prepare_data_for_prediction(self, df):
        """
        Veriyi model için hazırlar
        """
        try:
            if df is None or df.empty:
                return np.zeros((1, 60, 1))  # Boş veri durumunda sıfır matrisi döndür
            
            # Son 60 kapanış fiyatını al
            last_60_closes = df['close'].tail(60).values
            if len(last_60_closes) < 60:
                # Yeterli veri yoksa, mevcut veriyi 60'a tamamla
                padding = np.zeros(60 - len(last_60_closes))
                last_60_closes = np.concatenate([padding, last_60_closes])
            
            # Veriyi normalize et
            scaler = MinMaxScaler(feature_range=(0, 1))
            normalized_data = scaler.fit_transform(last_60_closes.reshape(-1, 1))
            
            # Model için şekillendir
            return normalized_data.reshape(1, -1, 1)
        
        except Exception as e:
            print(f"Veri hazırlama hatası: {str(e)}")
            return np.zeros((1, 60, 1))  # Hata durumunda sıfır matrisi döndür
    
    def start_trading(self, symbol='SOLUSDT'):
        """
        Geliştirilmiş trading sistemi
        """
        print(f"{symbol} için trading başlatılıyor...")
        
        while True:
            try:
                # Veri toplama (artık self.data_collector'ı kullan)
                market_info = self.data_collector.get_market_info(symbol)
                data = self.data_collector.get_multi_timeframe_data(symbol)
                
                # Trading kararları
                decisions = self.get_trading_decision(symbol, data)
                
                # Sonuçları göster
                self.display_analysis(symbol, market_info, decisions, data)
                
                # Açık pozisyonları kontrol et
                self.check_open_positions(symbol, market_info['son_fiyat'])
                
                # Yeni trade fırsatlarını değerlendir
                self.evaluate_trading_opportunity(symbol, decisions, market_info['son_fiyat'])
                
                # Trade geçmişini kaydet
                self.save_trade_history()
                
                # 5 dakika bekle
                time.sleep(300)
                
            except Exception as e:
                print(f"Hata: {e}")
                time.sleep(60)
                
    def display_analysis(self, symbol, market_info, decisions, data):
        """
        Detaylı analiz sonuçlarını gösterir
        """
        print("\n" + "="*50)
        print(f"Tarih: {datetime.now()}")
        print(f"Coin: {symbol}")
        
        print(f"\nNİHAİ KARAR:")
        final = decisions['final']
        print(f"Sinyal: {final['signal']}")
        print(f"Trend Yönü: {final['trend_yönü']}")
        print(f"Güven Skoru: {final['güven']}")
        
        print(f"\nPiyasa Bilgileri:")
        print(f"Son Fiyat: {market_info['son_fiyat']}")
        print(f"24s Değişim: %{market_info['günlük_değişim']}")
        print(f"24s Hacim: {market_info['günlük_hacim']}")
        
        for timeframe in ['4h', '1h', '15m']:
            if timeframe in decisions:
                decision = decisions[timeframe]
                print(f"\n{timeframe} Analiz:")
                print(f"Sinyal: {decision['signal']}")
                print(f"Ham Skor: {decision['score']}")
                print(f"Ağırlıklı Skor: {decision['weighted_score']}")
                print(f"Güven: {decision['güven']}")

    def check_open_positions(self, symbol, current_price):
        # Bu metodun içeriği, TradingBot sınıfının içinde olmalıdır
        # Bu örnekte, açık pozisyonları kontrol etmek için bir metod ekleyeceğiz
        # Bu metodun gerçek işlevi, açık pozisyonları kontrol etmek için tasarlanmıştır
        pass

    def evaluate_trading_opportunity(self, symbol, decisions, current_price):
        # Bu metodun içeriği, TradingBot sınıfının içinde olmalıdır
        # Bu örnekte, yeni trade fırsatlarını değerlendirmek için bir metod ekleyeceğiz
        # Bu metodun gerçek işlevi, yeni trade fırsatlarını değerlendirmek için tasarlanmıştır
        pass 