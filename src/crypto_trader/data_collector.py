import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta

class DataCollector:
    def __init__(self, timeframes=['15m', '1h', '4h']):
        self.exchange = ccxt.binance()
        self.timeframes = timeframes
    
    def fetch_historical_data(self, symbol, timeframe='1h', limit=1000):
        """
        Belirli bir zaman dilimi için kripto para verilerini çeker ve teknik indikatörleri hesaplar
        """
        try:
            # OHLCV verilerini çek
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if len(ohlcv) == 0:
                print(f"Veri bulunamadı: {symbol} {timeframe}")
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Teknik indikatörleri hesapla
            try:
                df = self.add_indicators(df)
                if df is None or len(df) < 60:  # En az 60 veri noktası gerekli
                    print(f"Yetersiz veri: {symbol} {timeframe}")
                    return None
            except Exception as e:
                print(f"İndikatör hesaplama hatası ({symbol} {timeframe}): {e}")
                return None
            
            return df
            
        except Exception as e:
            print(f"Veri çekme hatası ({symbol} {timeframe}): {e}")
            return None
    
    def add_indicators(self, df):
        """
        Genişletilmiş teknik indikatörler
        """
        try:
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            df['BB_middle'] = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + (std * 2)
            df['BB_lower'] = df['BB_middle'] - (std * 2)
            
            # EMA'lar
            df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()
            df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()
            
            # Stochastic RSI
            rsi = df['RSI']
            stoch_k = 100 * (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
            df['StochRSI_K'] = stoch_k
            df['StochRSI_D'] = stoch_k.rolling(3).mean()
            
            # ADX Hesaplama
            # True Range
            df['high_low'] = df['high'] - df['low']
            df['high_close'] = abs(df['high'] - df['close'].shift())
            df['low_close'] = abs(df['low'] - df['close'].shift())
            df['TR'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            
            # +DM ve -DM
            df['up_move'] = df['high'] - df['high'].shift()
            df['down_move'] = df['low'].shift() - df['low']
            
            df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
            df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
            
            # ADX hesaplama
            period = 14
            df['TR14'] = df['TR'].rolling(period).mean()
            df['plus_di14'] = 100 * (df['plus_dm'].rolling(period).mean() / df['TR14'])
            df['minus_di14'] = 100 * (df['minus_dm'].rolling(period).mean() / df['TR14'])
            df['DX'] = 100 * abs(df['plus_di14'] - df['minus_di14']) / (df['plus_di14'] + df['minus_di14'])
            df['ADX'] = df['DX'].rolling(period).mean()
            df['DMP'] = df['plus_di14']
            df['DMN'] = df['minus_di14']
            
            # Volatilite hesaplamaları
            df['daily_return'] = df['close'].pct_change()
            df['volatility'] = df['daily_return'].rolling(window=20).std() * np.sqrt(365)
            
            # Bollinger Bant Genişliği
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
            
            # Basit hareketli ortalamalar
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            
            # Gereksiz sütunları temizle
            columns_to_drop = ['high_low', 'high_close', 'low_close', 'TR', 'up_move', 
                              'down_move', 'plus_dm', 'minus_dm', 'TR14', 'plus_di14', 
                              'minus_di14', 'DX']
            df.drop(columns=columns_to_drop, inplace=True)
            
            # NaN değerleri temizle
            df.dropna(inplace=True)
            
            return df
            
        except Exception as e:
            print(f"İndikatör hesaplama hatası: {str(e)}")
            print(f"Hata oluşan indikatör için df şekli: {df.shape}")
            # Temel indikatörleri hesaplamaya çalış
            try:
                # Basit indikatörler
                df['SMA_20'] = df['close'].rolling(window=20).mean()
                df['SMA_50'] = df['close'].rolling(window=50).mean()
                df['daily_return'] = df['close'].pct_change()
                df['volatility'] = df['daily_return'].rolling(window=20).std() * np.sqrt(365)
                df.dropna(inplace=True)
            except Exception as inner_e:
                print(f"Temel indikatör hesaplama hatası: {str(inner_e)}")
            return df
    
    def get_multi_timeframe_data(self, symbol):
        """
        Çoklu zaman dilimi verisi topla
        """
        try:
            data = {}
            for timeframe in self.timeframes:
                df = self.fetch_historical_data(symbol, timeframe)
                if df is not None and not df.empty:
                    # Teknik indikatörleri hesapla
                    df = self.add_indicators(df)
                    data[timeframe] = df
                else:
                    print(f"Veri alınamadı: {symbol} - {timeframe}")
                
            if not data:
                print(f"Hiçbir zaman dilimi için veri alınamadı: {symbol}")
                return None
            
            return data
            
        except Exception as e:
            print(f"Veri toplama hatası ({symbol}): {e}")
            return None
    
    def get_market_info(self, symbol):
        """
        Coin hakkında temel bilgileri çeker
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'son_fiyat': ticker['last'],
                'günlük_değişim': ticker['percentage'],
                'günlük_hacim': ticker['quoteVolume'],
                'en_yüksek_24h': ticker['high'],
                'en_düşük_24h': ticker['low']
            }
        except Exception as e:
            print(f"Piyasa bilgisi çekme hatası: {e}")
            return None 