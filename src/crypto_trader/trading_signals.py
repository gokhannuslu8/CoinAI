import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from telegram_bot import TelegramNotifier
from adaptive_trader import AdaptiveTrader
import json

class SignalGenerator:
    def __init__(self):
        self.active_trades = {}  # Açık pozisyonları takip etmek için
        self.telegram = TelegramNotifier()
        self.last_signals = {}  # Son sinyalleri saklamak için
        self.adaptive_trader = AdaptiveTrader()  # Yeni eklenen
        self.telegram.set_signal_generator(self)  # TelegramNotifier'a referans ver
        self.last_signal_times = {}  # Son sinyal zamanlarını takip etmek için
        self.signal_cooldown = 4 * 3600  # 4 saat (saniye cinsinden)
        
    def analyze_signals(self, df, symbol, timeframe):
        try:
            # Mevcut değerler
            current_price = df['close'].iloc[-1]
            current_rsi = df['RSI'].iloc[-1]
            current_macd = df['MACD'].iloc[-1]
            current_hist = df['MACD_Hist'].iloc[-1]
            current_adx = df['ADX'].iloc[-1]
            
            # Hacim analizi
            volume_data = self.analyze_volume(df)
            
            # Trend belirleme
            trend = self.determine_trend(df)
            
            # İndikatörleri bir sözlükte topla
            indicators = {
                'trend': trend,
                'rsi': current_rsi,
                'macd': current_hist,
                'adx': current_adx,
                'volume': volume_data
            }
            
            # Aktif trade kontrolü
            if symbol in self.active_trades:
                entry_data = self.active_trades[symbol]
                entry_price = entry_data['price']
                
                # Kar/zarar hesapla
                profit_loss = ((current_price - entry_price) / entry_price) * 100
                
                # Stop loss kontrolünü daha sık yap (her mum için)
                stop_level = entry_data.get('stop_loss', entry_price * 0.98)  # %2 varsayılan
                
                if current_price <= stop_level:
                    exit_data = {
                        'timestamp': datetime.now(),
                        'price': current_price,
                        'reason': "Stop Loss tetiklendi",
                        'profit_loss': profit_loss
                    }
                    
                    # Stop loss bildirimi
                    self.telegram.send_message(f"""🚫 STOP LOSS - {symbol.replace('/USDT', '')}

İşlem: {'LONG' if entry_data['signal'] == 'AL' else 'SHORT'}
Giriş: {entry_price:.4f}
Çıkış: {current_price:.4f}
Zarar: %{abs(profit_loss):.2f}

⚠️ Stop Loss seviyesi tetiklendi!""")
                    
                    self.record_signal_result(entry_data, exit_data)
                    del self.active_trades[symbol]
                    return None

            # Son sinyal zamanını kontrol et
            current_time = datetime.now().timestamp()
            if symbol in self.last_signal_times:
                time_since_last_signal = (current_time - self.last_signal_times[symbol]) / 60
                if time_since_last_signal < 30:  # 30 dakikaya düşürdük
                    return None
            
            # Temel indikatörler
            df['RSI'] = self.calculate_rsi(df)
            macd, signal, hist = self.calculate_macd(df)
            df['MACD'] = macd
            df['MACD_Signal'] = signal
            df['MACD_Hist'] = hist
            df['ADX'] = self.calculate_adx(df)
            
            # Ek indikatörler
            # Bollinger Bantları
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = self.calculate_bollinger_bands(df)
            
            # Stokastik RSI
            df['StochRSI_K'], df['StochRSI_D'] = self.calculate_stoch_rsi(df)
            
            # Destek/Direnç seviyeleri
            support, resistance = self.find_support_resistance(df)
            
            # Erken uyarı sistemi için ek kontroller
            early_signal = False
            early_signal_reasons = []
            
            # Ani yükseliş kontrolü
            rapid_rise = False
            
            # Ani hacim ve fiyat artışı kontrolü
            volume_ma = df['volume'].rolling(20).mean()
            current_volume = df['volume'].iloc[-1]
            price_change = ((current_price - df['close'].iloc[-2])/df['close'].iloc[-2]*100)
            
            if current_volume > volume_ma.iloc[-1] * 2 and price_change > 2:
                rapid_rise = True
                early_signal_reasons.append("Ani hacim artışı ✅")
            
            if df['MACD_Hist'].iloc[-1] > 0 and df['MACD_Hist'].iloc[-2] < 0:
                early_signal_reasons.append("MACD kesişimi ✅")
            
            bb_width = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
            if bb_width.iloc[-1] < bb_width.iloc[-20:].mean() * 0.8 and df['close'].iloc[-1] > df['BB_middle'].iloc[-1]:
                early_signal_reasons.append("BB kırılımı ✅")
            
            # Ana sinyal analizi
            signal_type = None
            confidence = self.calculate_confidence_score(df, current_price, indicators)
            
            if trend == "Yukarı" and confidence >= 70:
                signal_type = "AL"
                
                # Ani yükseliş varsa ve trend onayı da varsa birleşik mesaj
                if rapid_rise:
                    reasons_text = "\n".join(f"• {reason}" for reason in early_signal_reasons)
                    message = f"""⚡️ ERKEN UYARI + YENİ SİNYAL - {symbol.replace('/USDT', '')} ({timeframe})

Sinyal: {signal_type}
Fiyat: {current_price:.4f}
Değişim: %{price_change:.2f}
Güven: %{confidence:.1f}

🔍 Tespit Edilen Sinyaller:
{reasons_text}

📊 Göstergeler:
RSI: {current_rsi:.1f}
MACD: {current_hist:.6f}
Trend: {trend}
ADX: {current_adx:.1f}
Hacim: {'Yüksek ✅' if volume_data['volume_surge'] else 'Normal ⚠️'}

⚠️ Risk Yönetimi:
Stop Loss: %2.0 ({current_price * 0.98:.4f})
Kar Al 1: %3.0 ({current_price * 1.03:.4f})
Kar Al 2: %5.0 ({current_price * 1.05:.4f})
Kar Al 3: %8.0 ({current_price * 1.08:.4f})"""

                else:  # Normal sinyal mesajı
                    message = f"""🔔 YENİ SİNYAL - {symbol.replace('/USDT', '')} ({timeframe})

Sinyal: {signal_type}
Fiyat: {current_price:.4f}
Güven: %{confidence:.1f}

📊 Göstergeler:
RSI: {current_rsi:.1f}
MACD: {current_hist:.6f}
Trend: {trend}
ADX: {current_adx:.1f}
Hacim: {'Yüksek ✅' if volume_data['volume_surge'] else 'Normal ⚠️'}

⚠️ Risk Yönetimi:
Stop Loss: %2.0 ({current_price * 0.98:.4f})
Kar Al 1: %3.0 ({current_price * 1.03:.4f})
Kar Al 2: %5.0 ({current_price * 1.05:.4f})
Kar Al 3: %8.0 ({current_price * 1.08:.4f})"""

                self.telegram.send_message(message)
                
                # Sinyal verilerini hazırla
                signal_data = {
                    "symbol": symbol,
                    "timestamp": datetime.now(),
                    "timeframe": timeframe,
                    "signal": signal_type,
                    "price": current_price,
                    "confidence": confidence,
                    "indicators": indicators
                }
                
                return signal_data

            return None

        except Exception as e:
            print(f"Sinyal analizi hatası: {str(e)}")
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

    def determine_trend(self, df):
        """
        Fiyat trendini belirler
        """
        try:
            # Son kapanış fiyatı
            last_close = df['close'].iloc[-1]
            
            # EMA'ları kontrol et
            if 'EMA_20' in df.columns and 'EMA_50' in df.columns:
                ema20 = df['EMA_20'].iloc[-1]
                ema50 = df['EMA_50'].iloc[-1]
                
                # Yukarı trend
                if last_close > ema20 > ema50:
                    return "Yukarı"
                # Aşağı trend
                elif last_close < ema20 < ema50:
                    return "Aşağı"
                # Yatay trend
                else:
                    # Son 20 mumun yönünü kontrol et
                    price_change = ((last_close - df['close'].iloc[-20]) / df['close'].iloc[-20]) * 100
                    if price_change > 2:
                        return "Yukarı"
                    elif price_change < -2:
                        return "Aşağı"
                    else:
                        return "Yatay"
                    
        except Exception as e:
            print(f"Trend belirleme hatası: {str(e)}")
            return "Belirsiz"

    def calculate_confidence_score(self, df, current_price, indicators):
        """
        Gerçek güven skoru hesaplar
        """
        try:
            confidence = 0
            conditions_met = 0
            
            # 1. Trend ve EMA Dizilimi (%30)
            if indicators['trend'] == "Yukarı":
                last_close = df['close'].iloc[-1]
                if 'EMA_20' in df.columns and 'EMA_50' in df.columns:
                    if last_close > df['EMA_20'].iloc[-1] > df['EMA_50'].iloc[-1]:
                        confidence += 30
                        conditions_met += 1
                    elif last_close > df['EMA_20'].iloc[-1]:
                        confidence += 15
            
            # 2. RSI + MACD Kombinasyonu (%25)
            rsi = indicators['rsi']
            if 40 <= rsi <= 60:
                if indicators['macd'] > 0 and df['MACD_Hist'].iloc[-1] > df['MACD_Hist'].iloc[-2]:
                    confidence += 25
                    conditions_met += 1
            
            # 3. ADX Trend Gücü (%20)
            if indicators['adx'] > 25:
                confidence += 20
                conditions_met += 1
            
            # 4. Hacim Analizi (%15)
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            if df['volume'].iloc[-1] > avg_volume:
                confidence += 15
                conditions_met += 1
            
            # 5. Bollinger Bantları (%10)
            if df['close'].iloc[-1] > df['BB_middle'].iloc[-1]:
                confidence += 10
                conditions_met += 1

            # Koşulları karşılama durumuna göre bonus
            if conditions_met >= 5:  # Tüm koşullar sağlandı
                return 100
            elif conditions_met == 4:  # 4 koşul sağlandı
                return min(95, confidence + 10)
            elif conditions_met == 3:  # 3 koşul sağlandı
                return min(85, confidence)
            else:
                return min(70, confidence)  # 2 veya daha az koşul
            
        except Exception as e:
            print(f"Güven skoru hesaplama hatası: {str(e)}")
            return 50

    def calculate_sat_confidence_score(self, df, current_price, indicators):
        """
        SAT sinyalleri için güven skoru hesaplar
        """
        try:
            confidence = 0
            conditions_met = 0
            
            # 1. Trend ve EMA Dizilimi (%30)
            if indicators['trend'] == "Aşağı":
                last_close = df['close'].iloc[-1]
                if 'EMA_20' in df.columns and 'EMA_50' in df.columns:
                    if last_close < df['EMA_20'].iloc[-1] < df['EMA_50'].iloc[-1]:
                        confidence += 30
                        conditions_met += 1
                    elif last_close < df['EMA_20'].iloc[-1]:
                        confidence += 15
            
            # 2. RSI + MACD Kombinasyonu (%25)
            rsi = indicators['rsi']
            if 60 <= rsi <= 80:
                if indicators['macd'] < 0 and df['MACD_Hist'].iloc[-1] < df['MACD_Hist'].iloc[-2]:
                    confidence += 25
                    conditions_met += 1
            
            # 3. ADX Trend Gücü (%20)
            if indicators['adx'] > 25:
                confidence += 20
                conditions_met += 1
            
            # 4. Hacim Analizi (%15)
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            if df['volume'].iloc[-1] > avg_volume:
                confidence += 15
                conditions_met += 1
            
            # 5. Bollinger Bantları (%10)
            if df['close'].iloc[-1] < df['BB_middle'].iloc[-1]:
                confidence += 10
                conditions_met += 1

            # Koşulları karşılama durumuna göre bonus
            if conditions_met >= 5:  # Tüm koşullar sağlandı
                return 100
            elif conditions_met == 4:  # 4 koşul sağlandı
                return min(95, confidence + 10)
            elif conditions_met == 3:  # 3 koşul sağlandı
                return min(85, confidence)
            else:
                return min(70, confidence)  # 2 veya daha az koşul
            
        except Exception as e:
            print(f"SAT güven skoru hesaplama hatası: {str(e)}")
            return 50

    def find_support_levels(self, df, window=20):
        """
        Destek seviyelerini bulur
        """
        try:
            levels = []
            for i in range(window, len(df)-window):
                if self.is_support(df, i):
                    levels.append(df['low'].iloc[i])
            return levels
        except Exception as e:
            print(f"Destek seviyesi hesaplama hatası: {str(e)}")
            return []

    def find_resistance_levels(self, df, window=20):
        """
        Direnç seviyelerini bulur
        """
        try:
            levels = []
            for i in range(window, len(df)-window):
                if self.is_resistance(df, i):
                    levels.append(df['high'].iloc[i])
            return levels
        except Exception as e:
            print(f"Direnç seviyesi hesaplama hatası: {str(e)}")
            return []

    def is_support(self, df, i):
        """Destek noktası kontrolü"""
        return (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                df['low'].iloc[i] < df['low'].iloc[i+1] and
                df['low'].iloc[i+1] < df['low'].iloc[i+2] and
                df['low'].iloc[i-1] < df['low'].iloc[i-2])

    def is_resistance(self, df, i):
        """Direnç noktası kontrolü"""
        return (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i+1] and
                df['high'].iloc[i+1] > df['high'].iloc[i+2] and
                df['high'].iloc[i-1] > df['high'].iloc[i-2])

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

    def check_exit_signals(self, df, entry_data):
        """
        Pozisyon çıkış kontrolü
        """
        try:
            entry_price = entry_data['price']
            current_price = df['close'].iloc[-1]
            entry_signal = entry_data['signal']
            
            # Kar/zarar hesapla
            if entry_signal == "AL":
                profit_loss = ((current_price - entry_price) / entry_price) * 100
            else:
                profit_loss = ((entry_price - current_price) / entry_price) * 100
            
            # Dinamik hedefleri kontrol et
            if 'targets' not in entry_data:
                entry_data['targets'] = self.calculate_dynamic_targets(df, entry_price, entry_data['symbol'])
            
            targets = entry_data['targets']
            
            # Stop loss kontrolü
            stop_level = entry_data.get('stop_loss', entry_price * (1 - targets['initial_stop']/100))
            if (entry_signal == "AL" and current_price <= stop_level) or \
               (entry_signal == "SAT" and current_price >= stop_level):
                return True, current_price, f"Stop Loss tetiklendi (-%{targets['initial_stop']})"
            
            # Kar Al hedefleri kontrolü
            if entry_signal == "AL":
                # 3. Hedef
                if profit_loss >= targets['tp3']:
                    return True, current_price, f"Kar Al 3 (%{targets['tp3']}) hedefine ulaşıldı"
                
                # 2. Hedef
                elif profit_loss >= targets['tp2'] and not entry_data.get('tp2_triggered'):
                    entry_data['tp2_triggered'] = True
                    entry_data['stop_loss'] = entry_price * (1 + targets['sl2']/100)
                    self.telegram.send_message(f"""🎯 KAR AL 2 - {entry_data['symbol'].replace('/USDT', '')}
Kar: %{profit_loss:.2f}
Stop Loss %{targets['sl2']}'e güncellendi""")
                
                # 1. Hedef
                elif profit_loss >= targets['tp1'] and not entry_data.get('tp1_triggered'):
                    entry_data['tp1_triggered'] = True
                    entry_data['stop_loss'] = entry_price * (1 + targets['sl1']/100)
                    self.telegram.send_message(f"""🎯 KAR AL 1 - {entry_data['symbol'].replace('/USDT', '')}
Kar: %{profit_loss:.2f}
Stop Loss %{targets['sl1']}'e güncellendi""")
            
            # Trend zayıflama kontrolü
            if self.is_trend_weakening(df, entry_signal):
                return True, current_price, "Trend zayıflaması tespit edildi"
            
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

    def find_support_resistance(self, df, period=20):
        """
        Destek ve direnç seviyeleri bul
        """
        try:
            # Pivot noktaları
            highs = df['high'].rolling(window=period, center=True).max()
            lows = df['low'].rolling(window=period, center=True).min()
            
            # Son fiyat
            current_price = df['close'].iloc[-1]
            
            # En yakın destek ve direnç
            supports = lows[lows < current_price].nlargest(3)
            resistances = highs[highs > current_price].nsmallest(3)
            
            if len(supports) > 0 and len(resistances) > 0:
                return supports.iloc[0], resistances.iloc[0]
            
            return None, None
            
        except Exception as e:
            print(f"Destek/Direnç hesaplama hatası: {str(e)}")
            return None, None

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

    def record_signal_result(self, entry_data, exit_data):
        """
        İşlem sonucunu kaydet ve bildirim gönder
        """
        try:
            trade_data = {
                'symbol': entry_data['symbol'],
                'entry_date': entry_data['timestamp'],
                'exit_date': exit_data['timestamp'],
                'signal_type': entry_data['signal'],
                'entry_price': entry_data['price'],
                'exit_price': exit_data['price'],
                'profit_loss': exit_data['profit_loss'],
                'confidence': entry_data.get('confidence', 0),
                'timeframe': entry_data['timeframe'],
                'indicators': entry_data.get('indicators', {}),
                'exit_reason': exit_data['reason']
            }
            
            # İşlem süresini hesapla
            duration = (exit_data['timestamp'] - entry_data['timestamp']).total_seconds() / 3600  # saat cinsinden
            
            # Kar/Zarar durumuna göre emoji seç
            if exit_data['profit_loss'] > 0:
                emoji = "✅"
            else:
                emoji = "❌"
            
            # Hedeflere göre başarı durumu
            targets = entry_data.get('targets', {})
            target_info = ""
            if targets:
                if exit_data['profit_loss'] >= targets.get('tp3', 8.0):
                    target_info = "3. Hedef başarıyla tamamlandı! 🎯🎯🎯"
                elif exit_data['profit_loss'] >= targets.get('tp2', 5.0):
                    target_info = "2. Hedefe ulaşıldı 🎯🎯"
                elif exit_data['profit_loss'] >= targets.get('tp1', 3.0):
                    target_info = "1. Hedefe ulaşıldı 🎯"
            
            # İşlem özet mesajı
            message = f"""📊 İŞLEM SONUCU {emoji}

Coin: {entry_data['symbol'].replace('/USDT', '')}
Timeframe: {entry_data['timeframe']}

📈 Giriş: {entry_data['price']:.4f}
📉 Çıkış: {exit_data['price']:.4f}
💰 Kar/Zarar: %{exit_data['profit_loss']:.2f}
⏱️ Süre: {duration:.1f} saat

🔍 Çıkış Nedeni: {exit_data['reason']}
{target_info}

📌 Özet:
- Giriş Güven: %{entry_data.get('confidence', 0):.1f}
- RSI: {entry_data.get('indicators', {}).get('rsi', 0):.1f}
- ADX: {entry_data.get('indicators', {}).get('adx', 0):.1f}"""

            # Telegram bildirimi gönder
            self.telegram.send_message(message)
            
            # AdaptiveTrader'a kaydet
            self.adaptive_trader.record_trade(trade_data)
            
        except Exception as e:
            print(f"İşlem sonucu kaydetme hatası: {str(e)}")

    def calculate_stoch_rsi(self, df, period=14, smoothK=3, smoothD=3):
        """
        Stochastic RSI hesapla
        """
        try:
            # Önce RSI hesapla
            rsi = df['RSI']
            
            # StochRSI = (RSI - RSI Low) / (RSI High - RSI Low)
            stochRSI = 100 * (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
            
            # %K ve %D hesapla
            K = stochRSI.rolling(smoothK).mean()
            D = K.rolling(smoothD).mean()
            
            return K, D
            
        except Exception as e:
            print(f"StochRSI hesaplama hatası: {str(e)}")
            return None, None

    def analyze_volume(self, df):
        """
        Hacim analizi yap
        """
        try:
            # Ortalama hacim (20 periyot)
            avg_volume = df['volume'].rolling(20).mean()
            current_volume = df['volume'].iloc[-1]
            
            # Hacim artış oranı
            volume_ratio = current_volume / avg_volume.iloc[-1]
            
            # Son 3 mumdaki hacim artışı
            volume_increasing = df['volume'].iloc[-3:].is_monotonic_increasing
            
            # Fiyat-hacim uyumu
            price_up = df['close'].iloc[-1] > df['close'].iloc[-2]
            volume_up = current_volume > df['volume'].iloc[-2]
            
            return {
                'avg_volume_ratio': volume_ratio,
                'volume_surge': volume_ratio > 1.5,  # Hacim patlaması
                'volume_confirms': (price_up and volume_up) or (not price_up and not volume_up),
                'volume_trend': "Artıyor" if volume_increasing else "Azalıyor"
            }
            
        except Exception as e:
            print(f"Hacim analizi hatası: {str(e)}")
            return {
                'avg_volume_ratio': 1.0,
                'volume_surge': False,
                'volume_confirms': False,
                'volume_trend': "Belirsiz"
            } 

    def calculate_dynamic_targets(self, df, entry_price, symbol):
        """
        Dinamik kar hedefleri hesapla
        """
        try:
            # Temel göstergeler
            adx = df['ADX'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # Volatilite hesaplama (son 20 mumdaki fiyat değişim yüzdesi)
            volatility = df['close'].pct_change().rolling(20).std().iloc[-1] * 100
            
            # Hacim analizi
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            
            # Baz hedefler (düşük volatilite durumu için)
            base_tp1 = 1.5  # İlk hedef
            base_tp2 = 2.5  # İkinci hedef
            base_tp3 = 4.0  # Üçüncü hedef
            
            # ADX'e göre trend gücü çarpanı
            if adx > 40:      # Çok güçlü trend
                trend_multiplier = 2.0
            elif adx > 30:    # Güçlü trend
                trend_multiplier = 1.5
            elif adx > 20:    # Normal trend
                trend_multiplier = 1.2
            else:            # Zayıf trend
                trend_multiplier = 1.0
            
            # Volatiliteye göre ayarlama
            if volatility > 5:     # Yüksek volatilite
                vol_multiplier = 1.5
            elif volatility > 3:   # Normal volatilite
                vol_multiplier = 1.2
            else:                  # Düşük volatilite
                vol_multiplier = 1.0
            
            # Hacim durumuna göre ayarlama
            if volume_ratio > 2.0:    # Çok yüksek hacim
                vol_boost = 1.3
            elif volume_ratio > 1.5:  # Yüksek hacim
                vol_boost = 1.2
            else:                     # Normal hacim
                vol_boost = 1.0
            
            # RSI durumuna göre ayarlama
            if rsi < 30:          # Aşırı satım
                rsi_multiplier = 1.3
            elif rsi > 70:        # Aşırı alım
                rsi_multiplier = 0.8
            else:                 # Normal
                rsi_multiplier = 1.0
            
            # Final hedefleri hesapla
            final_multiplier = trend_multiplier * vol_multiplier * vol_boost * rsi_multiplier
            
            targets = {
                'tp1': round(base_tp1 * final_multiplier, 1),
                'tp2': round(base_tp2 * final_multiplier, 1),
                'tp3': round(base_tp3 * final_multiplier, 1),
                'initial_stop': 2.0,  # Başlangıç stop loss
                'reason': f"ADX: {adx:.1f}, Volatilite: %{volatility:.1f}, Hacim: {volume_ratio:.1f}x"
            }
            
            # Stop loss seviyeleri
            targets['sl1'] = round(targets['tp1'] * 0.5, 1)  # İlk hedefe ulaşınca
            targets['sl2'] = round(targets['tp1'], 1)        # İkinci hedefe ulaşınca
            targets['sl3'] = round(targets['tp2'], 1)        # Üçüncü hedefe ulaşınca
            
            return targets
            
        except Exception as e:
            print(f"Dinamik hedef hesaplama hatası: {str(e)}")
            # Varsayılan hedefler
            return {
                'tp1': 3.0,
                'tp2': 5.0,
                'tp3': 8.0,
                'initial_stop': 2.0,
                'sl1': 1.5,
                'sl2': 3.0,
                'sl3': 5.0,
                'reason': "Varsayılan hedefler"
            } 

    def check_position_status(self, df, symbol):
        try:
            if symbol not in self.active_trades:
                return
            
            entry_data = self.active_trades[symbol]
            entry_price = entry_data['price']
            current_price = df['close'].iloc[-1]
            
            # Kar/zarar hesapla
            profit_loss = ((current_price - entry_price) / entry_price) * 100
            
            # Stop loss kontrolü
            stop_level = entry_price * 0.98  # %2 stop loss
            if current_price <= stop_level:
                message = f"""🚫 STOP LOSS - {symbol}

Giriş: {entry_price:.4f}
Çıkış: {current_price:.4f}
Zarar: %{abs(profit_loss):.2f}

Stop Loss seviyesi tetiklendi!"""
                
                self.telegram.send_message(message)
                
                # İşlemi kaydet
                trade_result = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol,
                    "signal_type": entry_data['signal'],
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "profit_loss": profit_loss,
                    "confidence": entry_data.get('confidence', 0),
                    "timeframe": entry_data.get('timeframe', '1h'),
                    "indicators": entry_data.get('indicators', {}),
                    "exit_reason": "Stop Loss tetiklendi"
                }
                
                # JSON dosyasına kaydet
                self._save_trade_result(trade_result)
                
                # Pozisyonu kapat
                del self.active_trades[symbol]
                return
            
            # Trend değişimi kontrolü
            if self._check_trend_reversal(df):
                message = f"""⚠️ TREND DEĞİŞİMİ - {symbol}

Giriş: {entry_price:.4f}
Mevcut: {current_price:.4f}
Kar/Zarar: %{profit_loss:.2f}

Trend tersine döndü, pozisyondan çıkılması önerilir."""
                
                self.telegram.send_message(message)
            
        except Exception as e:
            print(f"Pozisyon kontrol hatası: {str(e)}")

    def _save_trade_result(self, trade_result):
        """İşlem sonucunu JSON dosyasına kaydeder"""
        try:
            filename = f"trading_results/trades_{datetime.now().strftime('%Y%m%d')}.json"
            
            # Mevcut işlemleri oku
            try:
                with open(filename, 'r') as f:
                    trades = json.load(f)
            except:
                trades = []
            
            # Yeni işlemi ekle
            trades.append(trade_result)
            
            # Dosyaya kaydet
            with open(filename, 'w') as f:
                json.dump(trades, f, indent=4)
            
        except Exception as e:
            print(f"İşlem kaydetme hatası: {str(e)}")

    def _check_trend_reversal(self, df):
        """
        Trend değişimini daha hassas kontrol edelim
        """
        try:
            # Son 3 mumun EMA9 ve EMA21 ilişkisine bakalım
            ema9 = df['EMA_9'].iloc[-3:]
            ema21 = df['EMA_21'].iloc[-3:]
            
            if df['close'].iloc[-1] < ema9.iloc[-1] < ema21.iloc[-1]:
                return True
            
            return False
            
        except Exception as e:
            print(f"Trend kontrolü hatası: {str(e)}")
            return False

    def _handle_trend_reversal(self, symbol, entry_data, current_price, profit_loss):
        """
        Trend değişimi durumunda işlemleri yapar
        """
        message = f"""⚠️ TREND DEĞİŞİMİ - {symbol.replace('/USDT', '')}

İşlem: {'LONG' if entry_data['signal'] == 'AL' else 'SHORT'}
Giriş: {entry_data['price']:.4f}
Mevcut: {current_price:.4f}
Kar/Zarar: %{profit_loss:.2f}

❗️ Trend yön değiştirdi, pozisyondan çıkılması önerilir!"""

        self.telegram.send_message(message)

    def _close_position(self, symbol, reason, exit_price, profit_loss):
        """
        Pozisyonu kapatır ve sonucu kaydeder
        """
        entry_data = self.active_trades[symbol]
        exit_data = {
            'timestamp': datetime.now(),
            'price': exit_price,
            'reason': reason,
            'profit_loss': profit_loss
        }
        
        self.record_signal_result(entry_data, exit_data)
        del self.active_trades[symbol] 