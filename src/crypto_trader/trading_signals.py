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
        """
        Güçlü alım/satım sinyallerini tespit eder
        """
        # RSI kontrolü
        rsi_signal = 0
        current_rsi = df['RSI'].iloc[-1]
        prev_rsi = df['RSI'].iloc[-2]
        
        if current_rsi < 30 or (current_rsi < 35 and current_rsi > prev_rsi):
            rsi_signal = 1  # Aşırı satım - Alım fırsatı
        elif current_rsi > 70 or (current_rsi > 65 and current_rsi < prev_rsi):
            rsi_signal = -1  # Aşırı alım - Satım fırsatı
            
        # MACD kontrolü
        macd = df['MACD'].iloc[-1]
        signal = df['MACD_Signal'].iloc[-1]
        prev_macd = df['MACD'].iloc[-2]
        macd_signal = 1 if macd > signal and macd > prev_macd else -1 if macd < signal and macd < prev_macd else 0
        
        # Bollinger Bands kontrolü
        price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        upper_band = df['BB_upper'].iloc[-1]
        lower_band = df['BB_lower'].iloc[-1]
        bb_signal = 1 if price < lower_band or (price < lower_band * 1.02 and price > prev_price) else -1 if price > upper_band or (price > upper_band * 0.98 and price < prev_price) else 0
        
        # Trend kontrolü
        ma20 = df['close'].rolling(window=20).mean().iloc[-1]
        ma50 = df['close'].rolling(window=50).mean().iloc[-1]
        trend = 1 if price > ma20 and ma20 > ma50 else -1 if price < ma20 and ma20 < ma50 else 0
        
        # ADX değerini başta al
        adx = df['ADX'].iloc[-1]
        
        # Güçlü sinyal tespiti
        strong_signal = False
        signal_type = "BEKLE"
        confidence = 0
        
        # Alım sinyali koşulları
        if (rsi_signal == 1 and (macd_signal == 1 or bb_signal == 1)) or \
           (macd_signal == 1 and bb_signal == 1 and trend == 1):
            strong_signal = True
            signal_type = "AL"
            # Temel güven skoru
            confidence = 70  # Başlangıç skoru
            
            # Güven skoru artırıcıları
            if rsi_signal == 1:
                confidence += 5
            if macd_signal == 1:
                confidence += 5
            if bb_signal == 1:
                confidence += 5
            if trend == 1:
                confidence += 5
            if adx > 25:  # Güçlü trend
                confidence += 10
            
        # Satım sinyali koşulları
        elif (rsi_signal == -1 and (macd_signal == -1 or bb_signal == -1)) or \
             (macd_signal == -1 and bb_signal == -1 and trend == -1):
            strong_signal = True
            signal_type = "SAT"
            # Temel güven skoru
            confidence = 70  # Başlangıç skoru
            
            # Güven skoru artırıcıları
            if rsi_signal == -1:
                confidence += 5
            if macd_signal == -1:
                confidence += 5
            if bb_signal == -1:
                confidence += 5
            if trend == -1:
                confidence += 5
            if adx > 25:  # Güçlü trend
                confidence += 10
        
        # Güven skoru eşikleri - daha esnek
        MIN_CONFIDENCE_THRESHOLD = 70  # Minimum giriş için güven skoru (85'ten 70'e düşürdük)
        STRONG_CONFIDENCE_THRESHOLD = 85  # Çok güçlü sinyal eşiği (92'den 85'e düşürdük)
        
        # Güven skoruna göre sinyal gücünü belirle
        if confidence >= MIN_CONFIDENCE_THRESHOLD:
            strong_signal = True
            
            # Sinyal gücüne göre mesaj ekle
            if confidence >= STRONG_CONFIDENCE_THRESHOLD:
                signal_strength = "💎 ÇOK GÜÇLÜ"
                signal_data['signal_strength'] = "VERY_STRONG"
            else:
                signal_strength = "💪 GÜÇLÜ"
                signal_data['signal_strength'] = "STRONG"
            
            # Güven skoru detaylarını logla
            print(f"\nGüven Skoru Detayları - {symbol}:")
            print(f"RSI Skoru: {current_rsi:.1f}")
            print(f"MACD Skoru: {macd:.1f}")
            print(f"BB Skoru: {bb_signal:.1f}")
            print(f"Trend Skoru: {trend:.1f}")
            print(f"Toplam Baz Skor: {confidence:.1f}")
            print(f"Sinyal Gücü: {signal_strength}")
        else:
            strong_signal = False
            print(f"\n⚠️ Zayıf Sinyal - {symbol} (Güven: {confidence:.1f})")
        
        # Signal data güncelleme
        signal_data = {
            'timestamp': datetime.now(),
            'price': price,
            'signal': signal_type,
            'strong_signal': strong_signal,
            'confidence': confidence,
            'indicators': {
                'RSI': current_rsi,
                'MACD': macd,
                'MACD_Signal': signal,
                'BB_Upper': upper_band,
                'BB_Lower': lower_band,
                'MA20': ma20,
                'MA50': ma50,
                'ADX': adx,
                'Volume_Change': df['volume'].pct_change().iloc[-1]
            }
        }
        
        # Başlangıçta should_exit'i False olarak tanımla
        should_exit = False
        exit_price = None
        exit_reason = ""
        profit_loss = 0
        
        # Önce çıkış sinyallerini kontrol et
        if symbol in self.active_trades:
            should_exit, exit_price, exit_reason = self.check_exit_signals(df, self.active_trades[symbol]['signal'])
            
            if should_exit:
                entry_price = self.active_trades[symbol]['price']
                profit_loss = ((exit_price - entry_price) / entry_price) * 100 if self.active_trades[symbol]['signal'] == "AL" \
                             else ((entry_price - exit_price) / entry_price) * 100
                
                exit_data = {
                    'entry_price': entry_price,
                    'price': exit_price,
                    'profit_loss': profit_loss,
                    'message': exit_reason
                }
                
                # Çıkış sinyali gönder
                self.telegram.send_exit_signal(symbol, timeframe, exit_data)
                
                # Aktif işlemi kaldır
                del self.active_trades[symbol]
                
                # İşlem sonucunu adaptive trader'a gönder
                trade_data = {
                    'data': df,
                    'profit_loss': profit_loss,
                    'entry_signal': self.active_trades[symbol]['signal'],
                    'exit_reason': exit_reason,
                    'timeframe': timeframe
                }
                self.adaptive_trader.add_trade_result(trade_data)
        
        # Aktif işlem yoksa ve güçlü sinyal varsa giriş sinyali gönder
        if strong_signal and symbol not in self.active_trades:
            last_signal = self.last_signals.get(symbol, {})
            current_time = datetime.now()
            
            # Son sinyal zamanını kontrol et
            last_signal_time = last_signal.get('timestamp')
            min_time_between_signals = timedelta(hours=1)  # En az 1 saat bekle
            
            # Eğer son sinyal yoksa veya yeterli süre geçtiyse ve sinyal farklıysa
            if (not last_signal_time or 
                current_time - last_signal_time > min_time_between_signals) and \
                last_signal.get('signal') != signal_type:
                
                # Telegram bildirimi gönder
                self.telegram.send_signal(symbol, timeframe, signal_data)
                
                # Aktif işlemi kaydet
                self.active_trades[symbol] = {
                    'signal': signal_type,
                    'price': price,
                    'timestamp': current_time
                }
                
                # Son sinyali güncelle
                self.last_signals[symbol] = signal_data
        
        # Adaptif güven skoru hesapla
        confidence_boost = self.adaptive_trader.get_signal_confidence(df)
        
        if strong_signal:
            # Güven skorunu adaptif skorla güncelle
            confidence = confidence * confidence_boost
        
        return signal_data
        
    def check_exit_signals(self, df, entry_signal):
        """
        Çıkış sinyallerini kontrol eder
        """
        current_price = df['close'].iloc[-1]
        
        # Trend gücünü hesapla
        adx = df['ADX'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        macd_signal = df['MACD_Signal'].iloc[-1]
        
        # Dinamik kar hedefi hesapla
        if entry_signal == "AL":
            # Long pozisyon için çıkış sinyalleri
            entry_price = self.active_trades[symbol]['price']
            current_profit = ((current_price - entry_price) / entry_price) * 100
            
            # Trend zayıflama belirtileri
            trend_weakening = False
            
            # 1. RSI tepeden dönüş
            if rsi > 70 or (rsi > 65 and rsi < df['RSI'].iloc[-2]):
                trend_weakening = True
            
            # 2. MACD zayıflama
            if macd < macd_signal and macd < df['MACD'].iloc[-2]:
                trend_weakening = True
            
            # 3. ADX zayıflama
            if adx < 20 or adx < df['ADX'].iloc[-2]:
                trend_weakening = True
            
            # Kar hedefini dinamik ayarla
            if adx > 25:  # Güçlü trend
                target_profit = 5.0
            elif adx > 20:  # Orta trend
                target_profit = 3.0
            else:  # Zayıf trend
                target_profit = 2.0
            
            # Erken çıkış koşulları
            if current_profit >= target_profit:
                return True, current_price, f"Hedef kara ulaşıldı: %{current_profit:.2f}"
            
            elif current_profit >= target_profit * 0.7 and trend_weakening:
                return True, current_price, f"Trend zayıflıyor, erken çıkış: %{current_profit:.2f}"
            
            # Stop-loss kontrolü
            elif current_profit <= -2.0:  # %2 zarar
                return True, current_price, f"Stop-loss tetiklendi: %{current_profit:.2f}"
            
        elif entry_signal == "SAT":
            # Short pozisyon için çıkış sinyalleri
            entry_price = self.active_trades[symbol]['price']
            current_profit = ((entry_price - current_price) / entry_price) * 100
            
            # Trend zayıflama belirtileri
            trend_weakening = False
            
            # 1. RSI dipten dönüş
            if rsi < 30 or (rsi < 35 and rsi > df['RSI'].iloc[-2]):
                trend_weakening = True
            
            # 2. MACD zayıflama
            if macd > macd_signal and macd > df['MACD'].iloc[-2]:
                trend_weakening = True
            
            # 3. ADX zayıflama
            if adx < 20 or adx < df['ADX'].iloc[-2]:
                trend_weakening = True
            
            # Kar hedefini dinamik ayarla
            if adx > 25:  # Güçlü trend
                target_profit = 5.0
            elif adx > 20:  # Orta trend
                target_profit = 3.0
            else:  # Zayıf trend
                target_profit = 2.0
            
            # Erken çıkış koşulları
            if current_profit >= target_profit:
                return True, current_price, f"Hedef kara ulaşıldı: %{current_profit:.2f}"
            
            elif current_profit >= target_profit * 0.7 and trend_weakening:
                return True, current_price, f"Trend zayıflıyor, erken çıkış: %{current_profit:.2f}"
            
            # Stop-loss kontrolü
            elif current_profit <= -2.0:  # %2 zarar
                return True, current_price, f"Stop-loss tetiklendi: %{current_profit:.2f}"
            
        return False, current_price, "" 