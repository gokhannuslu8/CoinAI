import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from telegram_bot import TelegramNotifier
from adaptive_trader import AdaptiveTrader

class SignalGenerator:
    def __init__(self):
        self.active_trades = {}  # Açık pozisyonları takip etmek için
        self.telegram = TelegramNotifier()
        self.last_signals = {}  # Son sinyalleri saklamak için
        self.adaptive_trader = AdaptiveTrader()  # Yeni eklenen
        self.telegram.set_signal_generator(self)  # TelegramNotifier'a referans ver
        
    def analyze_signals(self, df, symbol, timeframe):
        try:
            # Aktif trade kontrolü
            if symbol in self.active_trades:
                entry_signal = self.active_trades[symbol]['signal']
                exit_signal, current_price, reason = self.check_exit_signals(df, entry_signal, symbol)
                
                if exit_signal:
                    exit_data = {
                        'timestamp': datetime.now(),
                        'price': current_price,
                        'reason': reason,
                        'profit_loss': profit_loss  # Kar/zarar bilgisi eklendi
                    }
                    
                    self.record_signal_result(self.active_trades[symbol], exit_data)
                    del self.active_trades[symbol]
                    
                    self.telegram.send_message(f"""⚠️ ÇIKIŞ SİNYALİ - {symbol.replace('/USDT', '')}

Sebep: {reason}
Fiyat: {current_price:.3f}
Kar/Zarar: %{profit_loss:.2f}""")
                    
                return None
                
            # RSI hesapla
            df['RSI'] = self.calculate_rsi(df)
            
            # MACD hesapla
            macd, signal, hist = self.calculate_macd(df)
            if macd is not None:
                df['MACD'] = macd
                df['MACD_Signal'] = signal
                df['MACD_Hist'] = hist
            else:
                return None
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df)
            df['BB_upper'] = bb_upper
            df['BB_middle'] = bb_middle
            df['BB_lower'] = bb_lower
            
            # ADX
            df['ADX'] = self.calculate_adx(df)
            
            # Değerleri al
            current_price = df['close'].iloc[-1]
            current_rsi = df['RSI'].iloc[-1]
            current_hist = df['MACD_Hist'].iloc[-1]
            current_adx = df['ADX'].iloc[-1]
            
            # Trend yönünü belirle
            trend = self.determine_trend(df, macd, signal)
            
            # Sinyal koşulları
            signal_type = self.generate_signal(
                trend=trend,
                rsi=current_rsi,
                macd_hist=current_hist,
                price=current_price,
                bb_lower=bb_lower.iloc[-1],
                bb_upper=bb_upper.iloc[-1],
                adx=current_adx
            )
            
            if signal_type:
                # Güven skorunu hesapla
                confidence = self.calculate_confidence(
                    trend=trend,
                    rsi=current_rsi,
                    macd_hist=current_hist,
                    adx=current_adx,
                    signal_type=signal_type
                )
                
                # Güven skoru %80'in altındaysa sinyal üretme
                if confidence < 80:  # 70'ten 80'e çıkarıldı
                    return None
                    
                # Sinyal verisi oluştur
                signal_data = {
                    "symbol": symbol,
                    "timestamp": datetime.now(),
                    "timeframe": timeframe,
                    "signal": signal_type,
                    "price": current_price,
                    "confidence": confidence,
                    "indicators": {
                        "rsi": current_rsi,
                        "macd": current_hist,
                        "trend": trend,
                        "adx": current_adx
                    }
                }
                
                # Aktif trade'lere ekle
                self.active_trades[symbol] = signal_data
                
                # Telegram mesajını oluştur ve gönder
                message = f"""💪 GÜÇLÜ SİNYAL - {symbol.replace('/USDT', '')} (1h)

Sinyal: {signal_type}
Fiyat: {current_price:.2f}
Güven: %{confidence:.1f}

📊 Göstergeler:
RSI: {current_rsi:.1f}
MACD: {"Yukarı" if current_hist > 0 else "Aşağı"}
Trend: {trend}
ADX: {current_adx:.1f}

🎯 Hedefler:
Stop Loss: %2.0
Kar Al: %3.0, %5.0, %8.0"""

                self.telegram.send_message(message)
                return signal_data
                
            return None
            
        except Exception as e:
            return None

    def format_signal_message(self, signal_data):
        """
        Sinyal mesajını formatla
        """
        symbol = signal_data["symbol"].replace("/", "")
        confidence = signal_data["adjusted_confidence"]
        model_boost = signal_data["model_boost"]
        stats = signal_data["statistics"]
        adx = signal_data["indicators"]["adx"]
        
        # Dinamik hedefleri hesapla
        initial_stop = 2.0  # Başlangıç stop-loss
        
        # ADX'e göre hedefleri ayarla
        if adx > 35:  # Çok güçlü trend
            target_levels = [3.0, 5.0, 8.0, 10.0]
            trail_levels = [1.2, 1.8, 2.4, 3.0]
        elif adx > 25:  # Güçlü trend
            target_levels = [2.5, 4.0, 6.0, 8.0]
            trail_levels = [1.0, 1.5, 2.0, 2.5]
        else:  # Normal trend
            target_levels = [2.0, 3.0, 4.0, 5.0]
            trail_levels = [0.8, 1.2, 1.6, 2.0]
        
        message = f"""💪 GÜÇLÜ SİNYAL - {symbol} ({signal_data['timeframe']})

Sinyal: {signal_data['signal']}
Fiyat: {signal_data['price']:.3f}
Güven: %{confidence:.1f} (Model boost: +{model_boost}%)

📊 Trend Analizi:
RSI: {signal_data['indicators']['rsi']:.2f}
MACD: {signal_data['indicators']['macd']['trend']}
BB: {signal_data['indicators']['bollinger']['position']}
Trend: {signal_data['indicators']['trend']}
ADX: {adx:.1f} (Güçlü Trend)

💰 Dinamik Hedefler:
İlk Stop: %{initial_stop:.1f}

🎯 Kar Alma Seviyeleri:
1️⃣ %{target_levels[0]:.1f} → Stop: %{trail_levels[0]:.1f}
2️⃣ %{target_levels[1]:.1f} → Stop: %{trail_levels[1]:.1f}
3️⃣ %{target_levels[2]:.1f} → Stop: %{trail_levels[2]:.1f}
4️⃣ %{target_levels[3]:.1f} → Stop: %{trail_levels[3]:.1f}

⚠️ Not: Trailing stop seviyeleri otomatik ayarlanacak

📈 Model İstatistikleri:
Toplam İşlem: {stats['total_trades']}
Başarı Oranı: %{stats['success_rate']:.1f}
Benzer Pattern Başarısı: %{stats['pattern_success']:.1f}
"""
        return message

    def generate_signal(self, trend, rsi, macd_hist, price, bb_lower, bb_upper, adx, df=None, symbol=None, timeframe=None):
        try:
            # ALIŞ Sinyali Koşulları
            if (
                trend == "Yukarı" and  # Trend yukarı olmalı
                rsi < 50 and
                macd_hist > 0 and  # MACD yukarı olmalı
                price < (bb_lower * 1.1)
            ):
                return "AL"
            
            # SATIŞ Sinyali Koşulları    
            elif (
                trend == "Aşağı" and  # Trend aşağı olmalı
                rsi > 55 and
                macd_hist < 0 and  # MACD aşağı olmalı
                price > (bb_upper * 0.9)
            ):
                return "SAT"
            
            return None
            
        except Exception as e:
            return None

    def determine_trend(self, df, macd, signal):
        """
        Trend belirleme - birden fazla göstergeye bakarak
        """
        # Son 20 mum için EMA hesapla
        ema20 = df['close'].ewm(span=20).mean()
        
        # Son değerleri al
        current_price = df['close'].iloc[-1]
        current_ema = ema20.iloc[-1]
        
        # MACD trend yönü
        macd_trend = "Yukarı" if macd.iloc[-1] > signal.iloc[-1] else "Aşağı"
        
        # Fiyat EMA üzerinde ve MACD yukarı ise güçlü yukarı trend
        if current_price > current_ema and macd_trend == "Yukarı":
            return "Yukarı"
        # Fiyat EMA altında ve MACD aşağı ise güçlü aşağı trend    
        elif current_price < current_ema and macd_trend == "Aşağı":
            return "Aşağı"
        else:
            return "Yatay"

    def calculate_confidence(self, trend, rsi, macd_hist, adx, signal_type):
        """
        Daha sıkı güven skoru hesaplama
        """
        confidence = 50  # Başlangıç değeri
        
        # Trend teyidi (daha yüksek ağırlık)
        if signal_type == "AL":
            if trend == "Yukarı":
                confidence += 20  # 15'ten 20'ye çıkarıldı
            elif trend == "Yatay":
                confidence += 10
        elif signal_type == "SAT":
            if trend == "Aşağı":
                confidence += 20
            elif trend == "Yatay":
                confidence += 10
            
        # RSI teyidi (daha sıkı koşullar)
        if signal_type == "AL":
            if 30 <= rsi <= 40:  # İdeal alım bölgesi
                confidence += 15
            elif 40 < rsi <= 45:  # Kabul edilebilir bölge
                confidence += 10
        elif signal_type == "SAT":
            if 60 <= rsi <= 70:  # İdeal satım bölgesi
                confidence += 15
            elif 55 <= rsi < 60:  # Kabul edilebilir bölge
                confidence += 10
            
        # MACD teyidi
        if signal_type == "AL" and macd_hist > 0:
            confidence += 15  # 10'dan 15'e çıkarıldı
        elif signal_type == "SAT" and macd_hist < 0:
            confidence += 15
            
        # ADX teyidi (trend gücü)
        if adx > 25:  # Güçlü trend
            confidence += 15
        if adx > 35:  # Çok güçlü trend
            confidence += 10
            
        return min(confidence, 95)  # Maximum 95% güven

    def calculate_rsi(self, df):
        """
        RSI (Relative Strength Index) hesaplama
        """
        try:
            # Fiyat değişimlerini hesapla
            delta = df['close'].diff()
            
            # Pozitif ve negatif değişimleri ayır
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            # RS ve RSI hesapla
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            print(f"RSI hesaplama hatası: {str(e)}")
            return pd.Series(index=df.index)  # Boş seri döndür

    def calculate_macd(self, df):
        """
        MACD hesaplama
        """
        try:
            # MACD hesaplama
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            hist = macd - signal
            
            return macd, signal, hist
            
        except Exception as e:
            print(f"MACD hesaplama hatası: {str(e)}")
            return None, None, None

    def calculate_bollinger_bands(self, df):
        """
        Bollinger Bands hesaplama
        """
        try:
            # Orta bant (20 günlük SMA)
            middle_band = df['close'].rolling(window=20).mean()
            
            # Standart sapma
            std = df['close'].rolling(window=20).std()
            
            # Üst ve alt bantlar
            upper_band = middle_band + (std * 2)
            lower_band = middle_band - (std * 2)
            
            return upper_band, middle_band, lower_band
            
        except Exception as e:
            print(f"Bollinger Bands hesaplama hatası: {str(e)}")
            return pd.Series(index=df.index), pd.Series(index=df.index), pd.Series(index=df.index)

    def calculate_adx(self, df):
        """
        ADX (Average Directional Index) hesaplama
        """
        try:
            # +DM ve -DM hesaplama
            high_diff = df['high'].diff()
            low_diff = df['low'].diff()
            
            pos_dm = high_diff.where((high_diff > 0) & (high_diff > low_diff.abs()), 0)
            neg_dm = low_diff.abs().where((low_diff < 0) & (low_diff.abs() > high_diff), 0)
            
            # True Range hesaplama
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # TR14, +DI14, -DI14 hesaplama
            tr14 = true_range.rolling(window=14).sum()
            pos_di14 = 100 * (pos_dm.rolling(window=14).sum() / tr14)
            neg_di14 = 100 * (neg_dm.rolling(window=14).sum() / tr14)
            
            # DX ve ADX hesaplama
            dx = 100 * ((pos_di14 - neg_di14).abs() / (pos_di14 + neg_di14))
            adx = dx.rolling(window=14).mean()
            
            return adx
            
        except Exception as e:
            print(f"ADX hesaplama hatası: {str(e)}")
            return pd.Series(index=df.index)

    def get_bb_position(self, price, upper_band, lower_band):
        # BB pozisyonu hesaplama işlemi
        # Bu işlemi gerçekleştirmek için gerekli kodu buraya ekleyin
        # Bu örnekte, BB pozisyonu hesaplama işlemi için basit bir mantık kullanılmıştır
        if price < lower_band:
            return "Altında"
        elif price > upper_band:
            return "Üstünde"
        else:
            return "Ortasında"

    def on_trade_complete(self, symbol, entry_signal, success):
        """
        İşlem tamamlandığında sonucu kaydet
        """
        if symbol in self.last_signals:
            pattern = self.last_signals[symbol]['indicators']
            self.adaptive_trader.record_trade_result(symbol, pattern, success)

    def check_exit_signals(self, df, entry_signal, symbol):
        try:
            entry_data = self.active_trades[symbol]
            entry_price = entry_data['price']
            current_price = df['close'].iloc[-1]
            
            # Kar/zarar hesapla
            if entry_signal == "AL":
                profit_loss = ((current_price - entry_price) / entry_price) * 100
            else:  # SAT sinyali
                profit_loss = ((entry_price - current_price) / entry_price) * 100
            
            # İndikatörleri hesapla
            df['RSI'] = self.calculate_rsi(df)
            macd, signal, hist = self.calculate_macd(df)
            df['MACD'] = macd
            df['MACD_Signal'] = signal
            df['MACD_Hist'] = hist  # Histogram değerini kaydediyoruz
            df['ADX'] = self.calculate_adx(df)
            
            current_rsi = df['RSI'].iloc[-1]
            current_macd = df['MACD'].iloc[-1]
            current_macd_signal = df['MACD_Signal'].iloc[-1]
            current_hist = df['MACD_Hist'].iloc[-1]  # Histogram değerini alıyoruz
            current_adx = df['ADX'].iloc[-1]

            # Giriş zamanından bu yana geçen süre (dakika)
            entry_time = entry_data['timestamp']
            time_in_position = (datetime.now() - entry_time).total_seconds() / 60
            
            # Stop Loss kontrolü (%2)
            if profit_loss < -2.0:
                if not entry_data.get('stop_loss_triggered', False):
                    exit_data = {
                        'timestamp': datetime.now(),
                        'price': current_price,
                        'reason': "Stop Loss tetiklendi",
                        'profit_loss': profit_loss
                    }
                    self.record_signal_result(entry_data, exit_data)
                    entry_data['stop_loss_triggered'] = True
                    
                    self.telegram.send_message(f"""🚫 STOP LOSS - {symbol.replace('/USDT', '')}
                    
Zarar: %{profit_loss:.2f}
Fiyat: {current_price:.4f}""")
                    
                    return True, current_price, "Stop Loss tetiklendi"
            
            # Trend zayıflama kontrolü - Daha sıkı koşullar
            if time_in_position > 30:  # En az 30 dakika pozisyonda kalalım
                weakness_points = 0
                
                # RSI kontrolü
                if entry_signal == "AL":
                    if current_rsi < 30:  # Aşırı satım
                        weakness_points += 1
                    if current_rsi < df['RSI'].iloc[-2] < df['RSI'].iloc[-3]:  # Düşüş trendi
                        weakness_points += 1
                else:
                    if current_rsi > 70:  # Aşırı alım
                        weakness_points += 1
                    if current_rsi > df['RSI'].iloc[-2] > df['RSI'].iloc[-3]:  # Yükseliş trendi
                        weakness_points += 1
                
                # MACD kontrolü
                if entry_signal == "AL":
                    if current_macd < current_macd_signal and current_hist < 0:  # current_hist kullanıyoruz
                        weakness_points += 1
                    if current_hist < df['MACD_Hist'].iloc[-2] < df['MACD_Hist'].iloc[-3]:
                        weakness_points += 1
                else:
                    if current_macd > current_macd_signal and current_hist > 0:
                        weakness_points += 1
                    if current_hist > df['MACD_Hist'].iloc[-2] > df['MACD_Hist'].iloc[-3]:
                        weakness_points += 1
                
                # ADX kontrolü
                if current_adx > 25:  # Güçlü trend varsa
                    if entry_signal == "AL" and df['DMN'].iloc[-1] > df['DMP'].iloc[-1]:
                        weakness_points += 2  # Güçlü aşağı trend
                    elif entry_signal == "SAT" and df['DMP'].iloc[-1] > df['DMN'].iloc[-1]:
                        weakness_points += 2  # Güçlü yukarı trend
                
                # Fiyat hareketi kontrolü
                last_candles = df['close'].tail(5)
                if entry_signal == "AL" and all(last_candles.iloc[i] > last_candles.iloc[i+1] for i in range(3)):
                    weakness_points += 1  # Son 4 mum düşüş
                elif entry_signal == "SAT" and all(last_candles.iloc[i] < last_candles.iloc[i+1] for i in range(3)):
                    weakness_points += 1  # Son 4 mum yükseliş
                
                # En az 4 zayıflama belirtisi varsa çık
                if weakness_points >= 4:
                    exit_data = {
                        'timestamp': datetime.now(),
                        'price': current_price,
                        'reason': "Güçlü trend değişimi",
                        'profit_loss': profit_loss
                    }
                    self.record_signal_result(entry_data, exit_data)
                    
                    self.telegram.send_message(f"""⚠️ TREND DEĞİŞİMİ - {symbol.replace('/USDT', '')}

Kar/Zarar: %{profit_loss:.2f}
Fiyat: {current_price:.4f}

📊 Göstergeler:
RSI: {current_rsi:.1f}
MACD: {"Yukarı" if current_macd > current_macd_signal else "Aşağı"}
ADX: {current_adx:.1f}

Pozisyondan çıkılıyor! ⛔️""")
                    
                    return True, current_price, "Güçlü trend değişimi"
            
            return False, current_price, None
            
        except Exception as e:
            print(f"Çıkış sinyali kontrolü hatası: {str(e)}")
            return False, None, None

    def calculate_trailing_stop(self, current_profit, highest_price=None, lowest_price=None, 
                              current_price=None, adx=None, position="LONG"):
        """
        Dinamik trailing stop hesaplama
        """
        if position == "LONG":
            # Kar yüzdesine göre trailing stop mesafesi
            if current_profit < 3.0:
                trail_percentage = 1.0  # %1 trailing stop
            elif current_profit < 5.0:
                trail_percentage = 1.5  # %1.5 trailing stop
            elif current_profit < 10.0:
                trail_percentage = 2.0  # %2 trailing stop
            else:
                trail_percentage = 2.5  # %2.5 trailing stop
            
            # ADX'e göre trailing stop'u ayarla
            if adx > 30:  # Güçlü trend
                trail_percentage *= 1.2  # Biraz daha fazla boşluk ver
            
            return highest_price * (1 - trail_percentage/100)
        
        else:  # SHORT pozisyon
            if current_profit < 3.0:
                trail_percentage = 1.0
            elif current_profit < 5.0:
                trail_percentage = 1.5
            elif current_profit < 10.0:
                trail_percentage = 2.0
            else:
                trail_percentage = 2.5
            
            if adx > 30:
                trail_percentage *= 1.2
            
            return lowest_price * (1 + trail_percentage/100)

    def detect_trend_weakness(self, rsi, macd, macd_signal, adx, df, position="LONG"):
        """
        Trend zayıflama belirtilerini tespit et
        """
        weakness_points = 0
        
        if position == "LONG":
            # RSI aşırı alım ve düşüş
            if rsi > 70 and rsi < df['RSI'].iloc[-2]:
                weakness_points += 1
            
            # MACD zayıflama
            if macd < macd_signal and macd < df['MACD'].iloc[-2]:
                weakness_points += 1
            
            # ADX zayıflama
            if adx < 20 or (adx < df['ADX'].iloc[-2] and adx < df['ADX'].iloc[-3]):
                weakness_points += 1
            
            # Fiyat momentum kaybı
            if df['close'].iloc[-1] < df['close'].iloc[-2] < df['close'].iloc[-3]:
                weakness_points += 1
            
        else:  # SHORT pozisyon
            # RSI aşırı satım ve yükseliş
            if rsi < 30 and rsi > df['RSI'].iloc[-2]:
                weakness_points += 1
            
            # MACD zayıflama
            if macd > macd_signal and macd > df['MACD'].iloc[-2]:
                weakness_points += 1
            
            # ADX zayıflama
            if adx < 20 or (adx < df['ADX'].iloc[-2] and adx < df['ADX'].iloc[-3]):
                weakness_points += 1
            
            # Fiyat momentum kaybı
            if df['close'].iloc[-1] > df['close'].iloc[-2] > df['close'].iloc[-3]:
                weakness_points += 1
            
        return weakness_points >= 2  # En az 2 zayıflama belirtisi varsa True 

    def send_position_update(self, symbol, current_price, current_profit, trailing_stop, trend_status):
        """
        Pozisyon güncellemelerini bildir
        """
        message = f"""📊 POZİSYON GÜNCELLEMESİ - {symbol}

💵 Mevcut Fiyat: {current_price:.3f}
📈 Kar/Zarar: %{current_profit:.2f}
🛡️ Trailing Stop: {trailing_stop:.3f}

⚡ Trend Durumu:
{trend_status}
"""
        self.telegram.send_message(message)

    def get_trend_status(self, rsi, macd, adx, df):
        if adx > 35:
            return "Çok Güçlü Trend"
        elif adx > 25:
            return "Güçlü Trend"
        else:
            return "Normal Trend"

    def analyze_volume_patterns(self, df):
        """
        Hacim paternlerini analiz eder
        """
        volume = df['volume']
        close = df['close']
        
        # Hacim ortalamaları
        vol_sma20 = volume.rolling(20).mean()
        vol_sma5 = volume.rolling(5).mean()
        
        # Hacim artışı kontrolü
        volume_surge = volume.iloc[-1] > vol_sma20.iloc[-1] * 1.5
        
        # Fiyat-Hacim uyumu
        price_up = close.iloc[-1] > close.iloc[-2]
        volume_up = volume.iloc[-1] > volume.iloc[-2]
        
        # Hacim teyidi
        volume_confirms = (price_up and volume_up) or (not price_up and not volume_up)
        
        return {
            'volume_surge': volume_surge,
            'volume_confirms': volume_confirms,
            'avg_volume_ratio': volume.iloc[-1] / vol_sma20.iloc[-1]
        }

    def find_support_resistance(self, df, lookback=100):
        """
        Önemli destek ve direnç seviyelerini belirler
        """
        highs = df['high'].iloc[-lookback:]
        lows = df['low'].iloc[-lookback:]
        
        # Pivot noktaları
        pivot_high = self.find_pivot_points(highs, 'high')
        pivot_low = self.find_pivot_points(lows, 'low')
        
        current_price = df['close'].iloc[-1]
        
        # En yakın seviyeleri bul
        nearest_support = max([p for p in pivot_low if p < current_price], default=None)
        nearest_resistance = min([p for p in pivot_high if p > current_price], default=None)
        
        return nearest_support, nearest_resistance

    def find_pivot_points(self, series, type='high'):
        """
        Pivot noktalarını belirler
        """
        pivot_points = []
        window = 5
        
        for i in range(window, len(series)-window):
            if type == 'high':
                if series.iloc[i] == max(series.iloc[i-window:i+window+1]):
                    pivot_points.append(series.iloc[i])
            else:
                if series.iloc[i] == min(series.iloc[i-window:i+window+1]):
                    pivot_points.append(series.iloc[i])
                
        return pivot_points

    def analyze_multiple_timeframes(self, symbol, current_timeframe):
        """
        Farklı zaman dilimlerinde analiz yapar
        """
        timeframes = {
            '15m': ['5m', '1h'],
            '1h': ['15m', '4h'],
            '4h': ['1h', '1d']
        }
        
        if current_timeframe not in timeframes:
            return None
        
        aligned_trends = 0
        results = {}
        
        for tf in timeframes[current_timeframe]:
            df = self.get_data_for_timeframe(symbol, tf)
            trend = self.determine_trend(df)
            results[tf] = trend
            
            if trend == self.determine_trend(df):
                aligned_trends += 1
            
        return {
            'timeframes': results,
            'alignment': aligned_trends / len(timeframes[current_timeframe])
        }

    def detect_candlestick_patterns(self, df):
        """
        Mum formasyonlarını tespit eder
        """
        patterns = {
            'hammer': self.is_hammer(df),
            'engulfing': self.is_engulfing(df),
            'doji': self.is_doji(df),
            'morning_star': self.is_morning_star(df),
            'evening_star': self.is_evening_star(df)
        }
        
        return patterns

    def is_hammer(self, df):
        """
        Çekiç formasyonu kontrolü
        """
        last_candle = df.iloc[-1]
        body = abs(last_candle['open'] - last_candle['close'])
        lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
        upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        
        return lower_wick > body * 2 and upper_wick < body * 0.5 

    def record_signal_result(self, signal_data, exit_data):
        """
        İşlem sonucunu kaydet
        """
        trade_data = {
            'symbol': signal_data['symbol'],
            'entry_date': signal_data['timestamp'],
            'exit_date': exit_data['timestamp'],
            'signal_type': signal_data['signal'],
            'entry_price': signal_data['price'],
            'exit_price': exit_data['price'],
            'profit_loss': exit_data['profit_loss'],
            'confidence': signal_data['confidence'],
            'timeframe': signal_data['timeframe'],
            'indicators': signal_data['indicators'],
            'exit_reason': exit_data['reason']
        }
        
        # AdaptiveTrader'a kaydet
        self.adaptive_trader.record_trade(trade_data) 