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
            # Teknik indikatörleri hesapla
            rsi = self.calculate_rsi(df)
            macd, signal, hist = self.calculate_macd(df)
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df)
            adx = self.calculate_adx(df)
            
            current_price = df['close'].iloc[-1]
            current_rsi = rsi.iloc[-1]
            current_hist = hist.iloc[-1]
            current_adx = adx.iloc[-1]
            
            # Trend yönünü belirle
            trend = self.determine_trend(df, macd, signal)
            
            # Sinyal koşulları - Düzeltilmiş mantık
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
                # Temel güven oranını hesapla
                confidence = self.calculate_confidence(
                    trend=trend,
                    rsi=current_rsi,
                    macd_hist=current_hist,
                    adx=current_adx,
                    signal_type=signal_type
                )
                
                # Model boost ve istatistikleri hesapla
                indicators = {
                    "rsi": current_rsi,
                    "macd": {
                        "histogram": current_hist,
                        "trend": "Yukarı kesişim" if current_hist > 0 else "Aşağı kesişim"
                    },
                    "bollinger": {
                        "position": self.get_bb_position(current_price, bb_upper.iloc[-1], bb_lower.iloc[-1]),
                        "width": (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]
                    },
                    "trend": trend,
                    "adx": current_adx
                }
                
                model_boost = self.adaptive_trader.calculate_model_boost(symbol, indicators)
                stats = self.adaptive_trader.get_trade_statistics(symbol)
                
                signal_data = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "signal": signal_type,
                    "price": current_price,
                    "confidence": confidence,
                    "model_boost": model_boost,
                    "adjusted_confidence": min(95, confidence + model_boost),
                    "indicators": indicators,
                    "statistics": stats
                }
                
                # Son sinyali sakla
                self.last_signals[symbol] = signal_data
                
                # Telegram bildirimi gönder
                message = self.format_signal_message(signal_data)
                self.telegram.send_message(message)
                
                return signal_data
                
        except Exception as e:
            print(f"Sinyal analiz hatası: {str(e)}")
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

    def generate_signal(self, trend, rsi, macd_hist, price, bb_lower, bb_upper, adx):
        """
        Geliştirilmiş sinyal üretme mantığı
        """
        # Hacim analizi
        volume_data = self.analyze_volume_patterns(df)
        
        # Destek/Direnç seviyeleri
        support, resistance = self.find_support_resistance(df)
        
        # Çoklu zaman dilimi analizi
        mtf_analysis = self.analyze_multiple_timeframes(symbol, timeframe)
        
        # Mum formasyonları
        patterns = self.detect_candlestick_patterns(df)
        
        # Güçlü trend kontrolü (ADX > 25)
        strong_trend = adx > 25
        
        # ALIŞ Sinyali için ek koşullar
        if (
            # Mevcut koşullar...
            (trend == "Yukarı" or not strong_trend) and
            rsi < 35 and rsi > 25 and
            macd_hist > 0 and
            price > bb_lower and price < (bb_lower * 1.02) and
            
            # Yeni koşullar
            volume_data['volume_confirms'] and
            volume_data['avg_volume_ratio'] > 1.2 and
            support is not None and price < support * 1.02 and
            mtf_analysis['alignment'] > 0.6 and
            (patterns['hammer'] or patterns['engulfing'])
        ):
            return "AL"
            
        # SATIŞ Sinyali Koşulları    
        elif (
            # Aşağı trend VEYA trend zayıf
            (trend == "Aşağı" or not strong_trend) and
            # RSI aşırı alım bölgesinden dönüyor
            rsi > 65 and rsi < 75 and
            # MACD histogramı negatife dönüyor
            macd_hist < 0 and
            # Fiyat üst BB'ye yakın ama kırmamış
            price < bb_upper and price > (bb_upper * 0.98)
        ):
            return "SAT"
            
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
        Sinyal güven oranı hesaplama - düzeltilmiş
        """
        confidence = 50  # Başlangıç değeri
        
        # Trend teyidi
        if signal_type == "AL" and trend == "Yukarı":
            confidence += 15
        elif signal_type == "SAT" and trend == "Aşağı":
            confidence += 15
            
        # RSI teyidi
        if signal_type == "AL" and 30 <= rsi <= 40:
            confidence += 10
        elif signal_type == "SAT" and 60 <= rsi <= 70:
            confidence += 10
            
        # MACD teyidi
        if (signal_type == "AL" and macd_hist > 0) or (signal_type == "SAT" and macd_hist < 0):
            confidence += 10
            
        # ADX teyidi
        if adx > 25:
            confidence += 10
        if adx > 35:
            confidence += 5
            
        return min(confidence, 95)  # Maximum 95% güven

    def calculate_rsi(self, df):
        # RSI hesaplama işlemi
        # Bu işlemi gerçekleştirmek için gerekli kodu buraya ekleyin
        # Bu örnekte, RSI hesaplama işlemi için pandas kullanılmıştır
        return df['RSI']

    def calculate_macd(self, df):
        # MACD hesaplama işlemi
        # Bu işlemi gerçekleştirmek için gerekli kodu buraya ekleyin
        # Bu örnekte, MACD hesaplama işlemi için pandas kullanılmıştır
        return df['MACD'], df['MACD_Signal'], df['MACD_Histogram']

    def calculate_bollinger_bands(self, df):
        # Bollinger Bands hesaplama işlemi
        # Bu işlemi gerçekleştirmek için gerekli kodu buraya ekleyin
        # Bu örnekte, Bollinger Bands hesaplama işlemi için pandas kullanılmıştır
        return df['BB_upper'], df['BB_middle'], df['BB_lower']

    def calculate_adx(self, df):
        # ADX hesaplama işlemi
        # Bu işlemi gerçekleştirmek için gerekli kodu buraya ekleyin
        # Bu örnekte, ADX hesaplama işlemi için pandas kullanılmıştır
        return df['ADX']

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
        """
        Dinamik çıkış sinyalleri ve trailing stop
        """
        try:
            current_price = df['close'].iloc[-1]
            entry_price = self.active_trades[symbol]['price']
            highest_price = self.active_trades[symbol].get('highest_price', entry_price)
            lowest_price = self.active_trades[symbol].get('lowest_price', entry_price)
            
            # Trend göstergeleri
            rsi = df['RSI'].iloc[-1]
            adx = df['ADX'].iloc[-1]
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_Signal'].iloc[-1]
            
            if entry_signal == "AL":
                # LONG pozisyon için
                current_profit = ((current_price - entry_price) / entry_price) * 100
                
                # En yüksek fiyatı güncelle
                if current_price > highest_price:
                    highest_price = current_price
                    self.active_trades[symbol]['highest_price'] = highest_price
                
                # Trailing stop hesapla
                trailing_stop = self.calculate_trailing_stop(
                    current_profit=current_profit,
                    highest_price=highest_price,
                    current_price=current_price,
                    adx=adx
                )
                
                # Trend zayıflama belirtileri
                trend_weakening = self.detect_trend_weakness(
                    rsi=rsi,
                    macd=macd,
                    macd_signal=macd_signal,
                    adx=adx,
                    df=df,
                    position="LONG"
                )
                
                # Çıkış koşulları
                if current_price <= trailing_stop:
                    trend_status = self.get_trend_status(rsi, macd, adx, df)
                    self.send_position_update(
                        symbol=symbol,
                        current_price=current_price,
                        current_profit=current_profit,
                        trailing_stop=trailing_stop,
                        trend_status=trend_status
                    )
                    return True, current_price, f"Trailing stop tetiklendi: %{current_profit:.2f}"
                    
                elif current_profit >= 5.0 and trend_weakening:
                    trend_status = self.get_trend_status(rsi, macd, adx, df)
                    self.send_position_update(
                        symbol=symbol,
                        current_price=current_price,
                        current_profit=current_profit,
                        trailing_stop=trailing_stop,
                        trend_status=trend_status
                    )
                    return True, current_price, f"Trend zayıflıyor, kar realizasyonu: %{current_profit:.2f}"
                    
                elif current_profit <= -2.0:  # Sabit stop-loss
                    trend_status = self.get_trend_status(rsi, macd, adx, df)
                    self.send_position_update(
                        symbol=symbol,
                        current_price=current_price,
                        current_profit=current_profit,
                        trailing_stop=trailing_stop,
                        trend_status=trend_status
                    )
                    return True, current_price, f"Stop-loss tetiklendi: %{current_profit:.2f}"
                    
            elif entry_signal == "SAT":
                # SHORT pozisyon için
                current_profit = ((entry_price - current_price) / entry_price) * 100
                
                # En düşük fiyatı güncelle
                if current_price < lowest_price:
                    lowest_price = current_price
                    self.active_trades[symbol]['lowest_price'] = lowest_price
                
                # Trailing stop hesapla
                trailing_stop = self.calculate_trailing_stop(
                    current_profit=current_profit,
                    lowest_price=lowest_price,
                    current_price=current_price,
                    adx=adx,
                    position="SHORT"
                )
                
                # Trend zayıflama belirtileri
                trend_weakening = self.detect_trend_weakness(
                    rsi=rsi,
                    macd=macd,
                    macd_signal=macd_signal,
                    adx=adx,
                    df=df,
                    position="SHORT"
                )
                
                # Çıkış koşulları
                if current_price >= trailing_stop:
                    trend_status = self.get_trend_status(rsi, macd, adx, df)
                    self.send_position_update(
                        symbol=symbol,
                        current_price=current_price,
                        current_profit=current_profit,
                        trailing_stop=trailing_stop,
                        trend_status=trend_status
                    )
                    return True, current_price, f"Trailing stop tetiklendi: %{current_profit:.2f}"
                    
                elif current_profit >= 5.0 and trend_weakening:
                    trend_status = self.get_trend_status(rsi, macd, adx, df)
                    self.send_position_update(
                        symbol=symbol,
                        current_price=current_price,
                        current_profit=current_profit,
                        trailing_stop=trailing_stop,
                        trend_status=trend_status
                    )
                    return True, current_price, f"Trend zayıflıyor, kar realizasyonu: %{current_profit:.2f}"
                    
                elif current_profit <= -2.0:  # Sabit stop-loss
                    trend_status = self.get_trend_status(rsi, macd, adx, df)
                    self.send_position_update(
                        symbol=symbol,
                        current_price=current_price,
                        current_profit=current_profit,
                        trailing_stop=trailing_stop,
                        trend_status=trend_status
                    )
                    return True, current_price, f"Stop-loss tetiklendi: %{current_profit:.2f}"
                    
            return False, current_price, ""
            
        except Exception as e:
            print(f"Çıkış sinyali kontrolü hatası: {str(e)}")
            return False, current_price, ""

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