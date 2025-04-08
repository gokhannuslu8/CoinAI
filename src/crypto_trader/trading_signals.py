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
            # Eğer coin zaten aktif işlemde ise, sadece pozisyon takibi yap
            if symbol in self.active_trades:
                self.check_position_status(df, symbol)
                return None
            
            # Son sinyal kontrolü - aynı coin için tekrar sinyal üretmeyi engelle
            current_time = datetime.now().timestamp()
            if symbol in self.last_signals:
                last_signal = self.last_signals[symbol]
                time_diff = current_time - last_signal['timestamp']
                
                # Son 4 saat içinde sinyal verildiyse tekrar verme
                if time_diff < 4 * 3600:  # 4 saat
                    return None

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
                
                # Son sinyali kaydet
                self.last_signals[symbol] = {
                    'timestamp': current_time,
                    'signal': signal_type,
                    'price': current_price
                }
                
                # Aktif işlemlere ekle
                self.active_trades[symbol] = {
                    'entry_price': current_price,
                    'signal': signal_type,
                    'entry_time': datetime.now(),
                    'timeframe': timeframe,
                    'stop_loss': current_price * 0.97,  # %3 stop
                    'take_profit1': current_price * 1.02,  # %2 kar
                    'take_profit2': current_price * 1.035,  # %3.5 kar
                    'take_profit3': current_price * 1.05,  # %5 kar
                    'tp1_hit': False,
                    'tp2_hit': False,
                    'tp3_hit': False
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
        try:
            confidence = 0
            conditions_met = 0
            
            # 1. Volatilite Kontrolü (%20)
            atr = self.calculate_atr(df)
            volatility = (atr / current_price) * 100
            
            if volatility <= 3:  # %3'ten düşük volatilite
                confidence += 20
                conditions_met += 1
            elif volatility > 5:  # Çok yüksek volatilite
                return 0  # Çok volatil piyasada işlem yapma
            
            # 2. Trend ve Momentum (%30)
            if indicators['trend'] == "Yukarı":
                # Son 3 mumun yönü
                last_3_candles = df.tail(3)
                if all(last_3_candles['close'] > last_3_candles['open']):
                    confidence += 30
                    conditions_met += 1
                
                # MACD kontrolü
                if df['MACD_Hist'].iloc[-1] > 0 and df['MACD_Hist'].iloc[-1] > df['MACD_Hist'].iloc[-2]:
                    confidence += 10
            
            # 3. Destek/Direnç Analizi (%25)
            support_levels = self.find_support_levels(df)
            nearest_support = min([abs(level - current_price) for level in support_levels])
            
            # Destek seviyesine yakınlık
            distance_to_support = (nearest_support / current_price) * 100
            if 0.5 <= distance_to_support <= 2:  # Destekten %0.5-%2 uzaklıkta
                confidence += 25
                conditions_met += 1
            
            # 4. Hacim Analizi (%25)
            volume_ma = df['volume'].rolling(20).mean()
            volume_trend = all(df['volume'].tail(3) > volume_ma.tail(3))
            
            if volume_trend and df['volume'].iloc[-1] > volume_ma.iloc[-1] * 1.5:
                confidence += 25
                conditions_met += 1
            
            # En az 3 koşul sağlanmalı ve toplam güven 85'in üzerinde olmalı
            return confidence if conditions_met >= 3 and confidence >= 85 else 0
            
        except Exception as e:
            print(f"Güven skoru hesaplama hatası: {str(e)}")
            return 0

    def calculate_atr(self, df, period=14):
        """Average True Range hesapla"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = abs(high - low)
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            
            return atr.iloc[-1]
        except Exception as e:
            print(f"ATR hesaplama hatası: {str(e)}")
            return None

    def find_support_levels(self, df, period=20):
        """Destek seviyelerini bul"""
        try:
            support_levels = []
            
            # Son period kadar mumu kontrol et
            for i in range(4, period):
                if (df['low'].iloc[-i] < df['low'].iloc[-i+1] and 
                    df['low'].iloc[-i] < df['low'].iloc[-i-1] and
                    df['low'].iloc[-i] < df['low'].iloc[-i+2]):
                    support_levels.append(df['low'].iloc[-i])
            
            return support_levels
        except Exception as e:
            print(f"Destek seviyesi hesaplama hatası: {str(e)}")
            return []

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
            
            position = self.active_trades[symbol]
            current_price = df['close'].iloc[-1]
            entry_price = position['entry_price']  # 'price' yerine 'entry_price' kullanıyoruz
            signal_type = position['signal']
            
            # Kar/zarar hesapla
            if signal_type == "AL":
                profit_loss = ((current_price - entry_price) / entry_price) * 100
                
                # Stop loss kontrolü
                if current_price <= position['stop_loss']:
                    message = f"""🚫 STOP LOSS - {symbol}

Giriş: {entry_price:.4f}
Çıkış: {current_price:.4f}
Zarar: %{abs(profit_loss):.2f}

Stop Loss seviyesi tetiklendi!"""
                    
                    self.telegram.send_message(message)
                    del self.active_trades[symbol]
                    return
                    
                # Kar hedefleri kontrolü
                if current_price >= position['take_profit1'] and not position['tp1_hit']:
                    message = f"""✅ KAR HEDEFİ 1 - {symbol}

Giriş: {entry_price:.4f}
Mevcut: {current_price:.4f}
Kar: %{profit_loss:.2f}

Stop-Loss seviyesi break-even'a çekildi."""
                    
                    self.telegram.send_message(message)
                    position['tp1_hit'] = True
                    position['stop_loss'] = entry_price  # Stop-loss'u giriş fiyatına çek
                    
                # Diğer kar hedefleri...
                
            else:  # SAT pozisyonu için
                profit_loss = ((entry_price - current_price) / entry_price) * 100
                
                # Stop loss kontrolü
                if current_price >= position['stop_loss']:
                    message = f"""🚫 STOP LOSS - {symbol}

Giriş: {entry_price:.4f}
Çıkış: {current_price:.4f}
Zarar: %{abs(profit_loss):.2f}

Stop Loss seviyesi tetiklendi!"""
                    
                    self.telegram.send_message(message)
                    del self.active_trades[symbol]
                    return
                    
                # Kar hedefleri kontrolü...
                
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
            return pd.Series([50] * len(df))  # Hata durumunda nötr değer döndür 

    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """
        MACD (Moving Average Convergence Divergence) hesaplar
        """
        try:
            # Hızlı ve yavaş EMA hesapla
            exp1 = df['close'].ewm(span=fast, adjust=False).mean()
            exp2 = df['close'].ewm(span=slow, adjust=False).mean()
            
            # MACD çizgisi
            macd = exp1 - exp2
            
            # Sinyal çizgisi
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            
            # Histogram
            histogram = macd - signal_line
            
            return macd, signal_line, histogram
            
        except Exception as e:
            print(f"MACD hesaplama hatası: {str(e)}")
            # Hata durumunda sıfır serisi döndür
            zeros = pd.Series([0] * len(df))
            return zeros, zeros, zeros 

    def calculate_adx(self, df, period=14):
        """
        ADX (Average Directional Index) hesaplar
        """
        try:
            # True Range hesapla
            df['TR'] = pd.DataFrame([
                df['high'] - df['low'],
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            ]).max()
            
            # +DM ve -DM hesapla
            df['HD'] = df['high'] - df['high'].shift(1)
            df['LD'] = df['low'].shift(1) - df['low']
            
            df['+DM'] = ((df['HD'] > df['LD']) & (df['HD'] > 0)) * df['HD']
            df['-DM'] = ((df['LD'] > df['HD']) & (df['LD'] > 0)) * df['LD']
            
            # ATR hesapla
            df['ATR'] = df['TR'].ewm(span=period, adjust=False).mean()
            
            # +DI ve -DI hesapla
            df['+DI'] = 100 * df['+DM'].ewm(span=period, adjust=False).mean() / df['ATR']
            df['-DI'] = 100 * df['-DM'].ewm(span=period, adjust=False).mean() / df['ATR']
            
            # DX ve ADX hesapla
            df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
            adx = df['DX'].ewm(span=period, adjust=False).mean()
            
            # Geçici sütunları temizle
            df.drop(['TR', 'HD', 'LD', '+DM', '-DM', 'ATR', '+DI', '-DI', 'DX'], axis=1, inplace=True)
            
            return adx
            
        except Exception as e:
            print(f"ADX hesaplama hatası: {str(e)}")
            return pd.Series([20] * len(df))  # Hata durumunda nötr değer döndür 

    def calculate_bollinger_bands(self, df, period=20, std_dev=2):
        """
        Bollinger Bands hesaplar
        """
        try:
            # Orta bant (20 periyot SMA)
            middle_band = df['close'].rolling(window=period).mean()
            
            # Standart sapma
            std = df['close'].rolling(window=period).std()
            
            # Üst ve alt bantlar
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)
            
            return upper_band, middle_band, lower_band
            
        except Exception as e:
            print(f"Bollinger Bands hesaplama hatası: {str(e)}")
            # Hata durumunda yaklaşık değerler döndür
            price = df['close'].iloc[-1]
            return (
                pd.Series([price * 1.02] * len(df)),  # Upper band
                pd.Series([price] * len(df)),         # Middle band
                pd.Series([price * 0.98] * len(df))   # Lower band
            ) 