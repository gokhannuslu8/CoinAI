import telebot
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
import pandas as pd
import numpy as np

class TelegramNotifier:
    def __init__(self):
        self.bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
        self.chat_id = TELEGRAM_CHAT_ID
        self.signal_generator = None  # SignalGenerator referansı
        
    def set_signal_generator(self, signal_generator):
        """SignalGenerator referansını ayarla"""
        self.signal_generator = signal_generator

    def send_message(self, message):
        try:
            self.bot.send_message(chat_id=self.chat_id, text=message)
            return True
        except Exception as e:
            print(f"Telegram mesaj hatası: {e}")
            return False
        
    def _calculate_success_rate(self):
        """Genel başarı oranını hesapla"""
        history = self.signal_generator.adaptive_trader.trade_history
        if len(history) == 0:
            return 0
        
        successful_trades = len(history[history['profit_loss'] > 0])
        return (successful_trades / len(history)) * 100
        
    def _calculate_pattern_success(self, current_signal):
        """Benzer pattern'lerin başarı oranını hesapla"""
        history = self.signal_generator.adaptive_trader.trade_history
        if len(history) < 10:  # Minimum 10 işlem olsun
            return 0
        
        # Benzer pattern'leri bul
        current_features = pd.DataFrame([current_signal['indicators']])
        similar_trades = 0
        successful_similar = 0
        
        for _, trade in history.iterrows():
            features = pd.DataFrame([trade['features']])
            similarity = self._calculate_similarity(current_features, features)
            
            if similarity > 0.8:  # %80 benzerlik
                similar_trades += 1
                if trade['profit_loss'] > 0:
                    successful_similar += 1
                
        return (successful_similar / max(1, similar_trades)) * 100
        
    def _calculate_similarity(self, features1, features2):
        """İki pattern arasındaki benzerliği hesapla"""
        diff = np.abs(features1 - features2)
        return 1 - (diff.mean().mean())

    def send_exit_signal(self, symbol, timeframe, signal_data):
        """
        Çıkış sinyalini Telegram'a gönderir
        """
        message = f"⚠️ ÇIKIŞ SİNYALİ - {symbol} ({timeframe})\n\n"
        
        if "profit_loss" in signal_data:
            message += f"Kar/Zarar: %{signal_data['profit_loss']:.2f}\n"
            
        if "message" in signal_data:
            message += f"Sebep: {signal_data['message']}\n"
            
        if "take_profit" in signal_data:
            message += f"Hedef: %{signal_data['take_profit']:.1f}\n"
            
        message += f"Fiyat: {signal_data['price']:.2f}"
        
        self.bot.send_message(self.chat_id, message)

    def send_test_message(self):
        """
        Test mesajı gönderir
        """
        try:
            message = "🔔 Test Mesajı!\n\n"
            message += "Trading Bot başarıyla çalışıyor.\n"
            message += "Bildirimler aktif ✅"
            
            self.bot.send_message(self.chat_id, message)
            return True
        except Exception as e:
            print(f"Telegram test mesajı hatası: {e}")
            return False 