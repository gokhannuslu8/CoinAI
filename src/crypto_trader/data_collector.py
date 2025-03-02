import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta

class DataCollector:
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'
            }
        })
    
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
        Tüm teknik indikatörleri hesaplar ve ekler
        """
        try:
            # RSI
            df['RSI'] = self.calculate_rsi(df)
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # EMA'lar
            df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()
            
            # Bollinger Bands
            df['BB_middle'] = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + (std * 2)
            df['BB_lower'] = df['BB_middle'] - (std * 2)
            
            # ADX
            df['ADX'] = self.calculate_adx(df)
            
            return df
            
        except Exception as e:
            print(f"İndikatör hesaplama hatası: {str(e)}")
            return df
    
    def get_current_price(self, symbol):
        """
        Anlık fiyat bilgisini getir
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
            
        except Exception as e:
            print(f"Fiyat alma hatası ({symbol}): {str(e)}")
            # Son kapanış fiyatını al
            data = self.get_multi_timeframe_data(symbol)
            if data and '1h' in data:
                return data['1h']['close'].iloc[-1]
            return None
            
    def get_multi_timeframe_data(self, symbol, timeframes=['15m', '1h', '4h']):
        """
        Geliştirilmiş çoklu zaman dilimi verisi toplama
        """
        try:
            data = {}
            for timeframe in timeframes:
                # Daha fazla veri noktası al
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=1000  # Daha fazla geçmiş veri
                )
                
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Temel indikatörler
                df = self.add_indicators(df)
                
                # Destek/Direnç seviyeleri
                df = self.add_support_resistance(df)
                
                # Hacim profili
                df = self.add_volume_profile(df)
                
                data[timeframe] = df
                
            return data
            
        except Exception as e:
            print(f"Veri alma hatası ({symbol}): {str(e)}")
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

    def add_support_resistance(self, df):
        """
        Destek ve direnç seviyelerini hesaplar
        """
        try:
            # Pivot noktaları
            high = df['high'].iloc[-1]
            low = df['low'].iloc[-1]
            close = df['close'].iloc[-1]
            
            pivot = (high + low + close) / 3
            
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)
            
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)
            
            df['Pivot'] = pivot
            df['R1'] = r1
            df['R2'] = r2
            df['R3'] = r3
            df['S1'] = s1
            df['S2'] = s2
            df['S3'] = s3
            
            return df
            
        except Exception as e:
            print(f"Destek/Direnç hesaplama hatası: {str(e)}")
            return df

    def add_volume_profile(self, df):
        """
        Hacim profili analizi ekler
        """
        try:
            # Son 100 mum için hacim profili
            price_bins = pd.qcut(df['close'].tail(100), q=10, labels=False)
            volume_profile = df['volume'].tail(100).groupby(price_bins).sum()
            
            # POC (Point of Control) - En yüksek hacimli seviye
            poc_index = volume_profile.idxmax()
            poc_price = df['close'].tail(100).iloc[poc_index]
            
            df['POC'] = poc_price
            df['Volume_Profile'] = volume_profile
            
            return df
            
        except Exception as e:
            print(f"Hacim profili hesaplama hatası: {str(e)}")
            return df

    def calculate_roc(self, series, period):
        """Rate of Change hesaplar"""
        return ((series - series.shift(period)) / series.shift(period)) * 100

    def calculate_mfi(self, df, period=14):
        """Money Flow Index hesaplar"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi

    def calculate_cmf(self, df, period=20):
        """Chaikin Money Flow hesaplar"""
        mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mf_volume = mf_multiplier * df['volume']
        
        return mf_volume.rolling(period).sum() / df['volume'].rolling(period).sum()

    def calculate_supertrend(self, df, period=10, multiplier=3):
        """
        SuperTrend indikatörünü hesaplar
        """
        try:
            # True Range hesaplama
            df['TR'] = pd.DataFrame([
                df['high'] - df['low'],
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            ]).max()
            
            # ATR (Average True Range) hesaplama
            df['ATR'] = df['TR'].rolling(period).mean()
            
            # SuperTrend hesaplama
            df['Upper Basic'] = (df['high'] + df['low']) / 2 + (multiplier * df['ATR'])
            df['Lower Basic'] = (df['high'] + df['low']) / 2 - (multiplier * df['ATR'])
            
            df['Upper'] = df['Upper Basic']
            df['Lower'] = df['Lower Basic']
            
            for i in range(1, len(df)):
                if df['close'].iloc[i-1] <= df['Upper'].iloc[i-1]:
                    df['Upper'].iloc[i] = min(df['Upper Basic'].iloc[i], df['Upper'].iloc[i-1])
                else:
                    df['Upper'].iloc[i] = df['Upper Basic'].iloc[i]
            
                if df['close'].iloc[i-1] >= df['Lower'].iloc[i-1]:
                    df['Lower'].iloc[i] = max(df['Lower Basic'].iloc[i], df['Lower'].iloc[i-1])
                else:
                    df['Lower'].iloc[i] = df['Lower Basic'].iloc[i]
            
            # SuperTrend değeri
            df['SuperTrend'] = np.nan
            for i in range(1, len(df)):
                if df['close'].iloc[i] <= df['Upper'].iloc[i]:
                    df['SuperTrend'].iloc[i] = df['Upper'].iloc[i]
                else:
                    df['SuperTrend'].iloc[i] = df['Lower'].iloc[i]
            
            # Gereksiz sütunları temizle
            df.drop(['TR', 'Upper Basic', 'Lower Basic', 'Upper', 'Lower'], axis=1, inplace=True)
            
            return df['SuperTrend']
            
        except Exception as e:
            print(f"SuperTrend hesaplama hatası: {str(e)}")
            return pd.Series(np.nan, index=df.index)

    def calculate_ichimoku(self, df):
        """
        Ichimoku indikatörünü hesaplar
        """
        try:
            # Tenkan-sen (Conversion Line)
            period9_high = df['high'].rolling(window=9).max()
            period9_low = df['low'].rolling(window=9).min()
            df['tenkan_sen'] = (period9_high + period9_low) / 2
            
            # Kijun-sen (Base Line)
            period26_high = df['high'].rolling(window=26).max()
            period26_low = df['low'].rolling(window=26).min()
            df['kijun_sen'] = (period26_high + period26_low) / 2
            
            # Senkou Span A (Leading Span A)
            df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
            
            # Senkou Span B (Leading Span B)
            period52_high = df['high'].rolling(window=52).max()
            period52_low = df['low'].rolling(window=52).min()
            df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
            
            # Chikou Span (Lagging Span)
            df['chikou_span'] = df['close'].shift(-26)
            
            return df['senkou_span_a']  # Ana sinyal çizgisini döndür
            
        except Exception as e:
            print(f"Ichimoku hesaplama hatası: {str(e)}")
            return pd.Series(np.nan, index=df.index)

    def calculate_keltner_channels(self, df, period=20, multiplier=2):
        """
        Keltner Channels hesaplar
        """
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            ema = typical_price.ewm(span=period, adjust=False).mean()
            atr = self.calculate_atr(df, period)
            
            kc_upper = ema + (multiplier * atr)
            kc_lower = ema - (multiplier * atr)
            
            return kc_upper, kc_lower
            
        except Exception as e:
            print(f"Keltner Channels hesaplama hatası: {str(e)}")
            return pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)

    def calculate_atr(self, df, period=14):
        """
        Average True Range hesaplar
        """
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            
            return true_range.rolling(period).mean()
            
        except Exception as e:
            print(f"ATR hesaplama hatası: {str(e)}")
            return pd.Series(np.nan, index=df.index)

    def calculate_obv(self, df):
        """
        On Balance Volume (OBV) hesaplar
        """
        try:
            obv = pd.Series(index=df.index, dtype='float64')
            obv.iloc[0] = 0
            
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            return obv
            
        except Exception as e:
            print(f"OBV hesaplama hatası: {str(e)}")
            return pd.Series(np.nan, index=df.index)

    def calculate_vwap(self, df):
        """
        Volume Weighted Average Price (VWAP) hesaplar
        """
        try:
            # Tipik fiyat = (Yüksek + Düşük + Kapanış) / 3
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            
            # Tipik fiyat * Hacim
            price_volume = typical_price * df['volume']
            
            # Kümülatif toplamlar
            cum_price_volume = price_volume.cumsum()
            cum_volume = df['volume'].cumsum()
            
            # VWAP = Kümülatif (Fiyat * Hacim) / Kümülatif Hacim
            vwap = cum_price_volume / cum_volume
            
            return vwap
            
        except Exception as e:
            print(f"VWAP hesaplama hatası: {str(e)}")
            return pd.Series(np.nan, index=df.index)

    def calculate_rsi(self, df, period=14):
        """
        RSI (Relative Strength Index) hesaplar
        """
        try:
            # Fiyat değişimlerini hesapla
            delta = df['close'].diff()
            
            # Pozitif ve negatif değişimleri ayır
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # RS = Ortalama Kazanç / Ortalama Kayıp
            rs = gain / loss
            
            # RSI = 100 - (100 / (1 + RS))
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            print(f"RSI hesaplama hatası: {str(e)}")
            return pd.Series(np.nan, index=df.index)

    def calculate_adx(self, df, period=14):
        """
        ADX (Average Directional Index) hesaplar
        """
        try:
            # True Range
            df['TR'] = pd.DataFrame([
                df['high'] - df['low'],
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            ]).max()
            
            # +DM ve -DM
            df['plus_DM'] = np.where(
                (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                np.maximum(df['high'] - df['high'].shift(1), 0),
                0
            )
            
            df['minus_DM'] = np.where(
                (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                np.maximum(df['low'].shift(1) - df['low'], 0),
                0
            )
            
            # TR14, +DM14, -DM14
            df['TR14'] = df['TR'].rolling(window=period).mean()
            df['plus_DM14'] = df['plus_DM'].rolling(window=period).mean()
            df['minus_DM14'] = df['minus_DM'].rolling(window=period).mean()
            
            # +DI14 ve -DI14
            df['plus_DI14'] = 100 * df['plus_DM14'] / df['TR14']
            df['minus_DI14'] = 100 * df['minus_DM14'] / df['TR14']
            
            # DX ve ADX
            df['DX'] = 100 * abs(df['plus_DI14'] - df['minus_DI14']) / (df['plus_DI14'] + df['minus_DI14'])
            adx = df['DX'].rolling(window=period).mean()
            
            # Gereksiz sütunları temizle
            df.drop(['TR', 'plus_DM', 'minus_DM', 'TR14', 'plus_DM14', 'minus_DM14', 
                    'plus_DI14', 'minus_DI14', 'DX'], axis=1, inplace=True)
            
            return adx
            
        except Exception as e:
            print(f"ADX hesaplama hatası: {str(e)}")
            return pd.Series(np.nan, index=df.index)

    def analyze_volume_patterns(self, df):
        """
        Hacim paternlerini analiz eder
        """
        try:
            # Hacim ortalamaları
            volume_ma = df['volume'].rolling(20).mean()
            volume_std = df['volume'].rolling(20).std()
            
            # Son 3 mumdaki hacim artışı
            recent_volume_increase = (df['volume'].iloc[-1] > volume_ma.iloc[-1] * 1.5 and
                                    df['volume'].iloc[-1] > df['volume'].iloc[-2] * 1.2)
            
            # Hacim trendini belirle
            volume_trend = "Yükseliyor" if df['volume'].iloc[-3:].mean() > volume_ma.iloc[-3:].mean() else "Normal"
            
            return {
                'volume_surge': recent_volume_increase,
                'volume_trend': volume_trend,
                'avg_volume': volume_ma.iloc[-1],
                'volume_std': volume_std.iloc[-1]
            }
            
        except Exception as e:
            print(f"Hacim analizi hatası: {str(e)}")
            return {} 