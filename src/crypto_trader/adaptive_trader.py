from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

class AdaptiveTrader:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.trade_history = pd.DataFrame(columns=[
            'symbol',
            'entry_date',
            'exit_date', 
            'signal_type',  # AL/SAT
            'entry_price',
            'exit_price',
            'profit_loss',  # Yüzdelik kar/zarar
            'confidence',   # Sinyal güven skoru
            'timeframe',
            'indicators',   # Girişteki indikatör değerleri
            'exit_reason'   # stop/target/trend_change
        ])
        self.pattern_history = {}  # Benzer pattern'ların başarı oranı
        self.min_trades_for_stats = 10  # İstatistik için minimum işlem sayısı
        self.min_samples = 50  # Minimum eğitim örneği sayısı
        
        # Trading results klasörünü oluştur
        self.results_dir = "trading_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        # Sabit dosya adı kullan
        self.results_file = f"{self.results_dir}/trading_history.json"
        
        # Dosyayı yükle veya oluştur
        self.load_trade_history()
        
    def prepare_features(self, data):
        """İndikatörlerden özellikler oluştur"""
        return {
            'rsi': data['RSI'].iloc[-1],
            'macd': data['MACD'].iloc[-1],
            'macd_signal': data['MACD_Signal'].iloc[-1],
            'bb_position': (data['close'].iloc[-1] - data['BB_lower'].iloc[-1]) / (data['BB_upper'].iloc[-1] - data['BB_lower'].iloc[-1]),
            'trend': 1 if data['close'].iloc[-1] > data['MA20'].iloc[-1] else -1,
            'volume_change': data['volume'].pct_change().iloc[-1],
            'price_change': data['close'].pct_change().iloc[-1]
        }
        
    def add_trade_result(self, trade_data):
        """
        Yeni işlem sonucunu kaydet
        """
        trade_result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': trade_data.get('symbol'),
            'timeframe': trade_data.get('timeframe'),
            'entry_price': trade_data.get('entry_price'),
            'exit_price': trade_data.get('price'),
            'profit_loss': trade_data.get('profit_loss'),
            'entry_signal': trade_data.get('entry_signal'),
            'exit_reason': trade_data.get('exit_reason'),
            'indicators': {
                'RSI': trade_data['data']['RSI'].iloc[-1],
                'MACD': trade_data['data']['MACD'].iloc[-1],
                'ADX': trade_data['data']['ADX'].iloc[-1]
            }
        }
        
        self.trade_history.append(trade_result)
        self.save_trade_history()
        
        print(f"\n💾 İşlem kaydedildi - {trade_result['symbol']}")
        print(f"Kar/Zarar: %{trade_result['profit_loss']:.2f}")
        
    def _update_model(self):
        """Modeli güncelle"""
        if len(self.trade_history) > self.min_samples:
            X = pd.DataFrame(self.trade_history['features'].tolist())
            y = self.trade_history['result']
            self.model.fit(X, y)
            
    def get_signal_confidence(self, current_data):
        """Sinyal güvenilirliğini hesapla"""
        if len(self.trade_history) < self.min_samples:
            return 1.0  # Yeterli veri yoksa normal güven skorunu kullan
            
        features = self.prepare_features(current_data)
        X = pd.DataFrame([features])
        
        # Modelin tahmin olasılıkları
        proba = self.model.predict_proba(X)[0]
        confidence = proba[1]  # Karlı işlem olasılığı
        
        # 0.5-1.0 arasını 0.8-1.2 aralığına dönüştür
        scaled_confidence = 0.8 + (confidence - 0.5) * 0.8
        return max(0.5, min(1.2, scaled_confidence))

    def save_trade_history(self):
        """İşlem geçmişini JSON dosyasına kaydet"""
        with open(self.results_file, 'w') as f:
            json.dump(self.trade_history, f, indent=4)

    def load_trade_history(self):
        """İşlem geçmişini JSON dosyasından yükle"""
        try:
            with open(self.results_file, 'r') as f:
                self.trade_history = json.load(f)
        except:
            self.trade_history = [] 

    def calculate_model_boost(self, symbol, indicators):
        try:
            # Başarı oranını al
            success_rate = self.get_symbol_success_rate(symbol)  # Bu fonksiyon eksik
            
            # Model boost hesapla
            boost = min(15, success_rate * 0.2)  # Maximum %15 boost
            return boost
            
        except Exception as e:
            print(f"Model boost hesaplama hatası: {str(e)}")
            return 0
            
    def create_pattern(self, indicators):
        """
        Mevcut market durumundan bir pattern oluşturur
        """
        return {
            'rsi_zone': self.get_rsi_zone(indicators['rsi']),
            'trend': indicators['trend'],
            'adx_strength': self.get_adx_strength(indicators['adx']),
            'bb_position': indicators['bollinger']['position'],
            'macd_trend': indicators['macd']['trend']
        }
        
    def record_trade_result(self, symbol, pattern, success):
        """
        İşlem sonucunu kaydeder
        """
        # Sembol bazlı işlem geçmişi
        if symbol not in self.trade_history:
            self.trade_history[symbol] = []
        self.trade_history[symbol].append(success)
        
        # Pattern bazlı işlem geçmişi
        pattern_key = str(pattern)
        if pattern_key not in self.pattern_history:
            self.pattern_history[pattern_key] = []
        self.pattern_history[pattern_key].append(success)
        
    def get_trade_statistics(self, symbol):
        """
        Sembol için istatistikleri getir
        """
        return {
            "total_trades": len(self.trade_history),
            "success_rate": 0,  # Henüz işlem yok
            "pattern_success": 0  # Henüz pattern analizi yok
        }
        
    def get_pattern_success_rate(self, pattern):
        """
        Benzer pattern'ların başarı oranını hesaplar
        """
        pattern_key = str(pattern)
        if pattern_key in self.pattern_history:
            results = self.pattern_history[pattern_key]
            if len(results) >= 5:  # En az 5 benzer işlem
                return sum(results) / len(results)
        return 0.5  # Varsayılan oran
        
    @staticmethod
    def get_rsi_zone(rsi):
        if rsi < 30: return 'oversold'
        if rsi > 70: return 'overbought'
        return 'neutral'
        
    @staticmethod
    def get_adx_strength(adx):
        if adx > 35: return 'strong'
        if adx > 25: return 'moderate'
        return 'weak'

    def get_symbol_success_rate(self, symbol):
        """
        Sembol için başarı oranını hesapla
        """
        if len(self.trade_history) == 0:
            return 0
        
        symbol_trades = self.trade_history[self.trade_history['symbol'] == symbol]
        if len(symbol_trades) == 0:
            return 0
        
        successful = len(symbol_trades[symbol_trades['profit'] > 0])
        return (successful / len(symbol_trades)) * 100 

    def record_trade(self, trade_data):
        """
        İşlem sonucunu kaydet
        """
        try:
            # Mevcut kayıtları yükle
            trades = self.load_trade_history()
            
            # Yeni işlemi ekle
            trades.append({
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': trade_data['symbol'],
                'signal_type': trade_data['signal_type'],
                'entry_price': trade_data['entry_price'],
                'exit_price': trade_data['exit_price'],
                'profit_loss': trade_data['profit_loss'],
                'confidence': trade_data['confidence'],
                'timeframe': trade_data['timeframe'],
                'indicators': trade_data['indicators'],
                'exit_reason': trade_data['exit_reason']
            })
            
            # Dosyaya kaydet
            with open(self.results_file, 'w') as f:
                json.dump(trades, f, indent=4)
            
            print(f"İşlem kaydedildi: {trade_data['symbol']} - {trade_data['profit_loss']:.2f}%")
            
            # İstatistikleri güncelle
            self.update_statistics(trade_data['symbol'])
            
        except Exception as e:
            print(f"İşlem kayıt hatası: {str(e)}")
            
    def update_statistics(self, symbol):
        """
        Symbol bazlı istatistikleri güncelle
        """
        try:
            # Trade history bir liste olduğu için önce DataFrame'e çevirelim
            df = pd.DataFrame(self.trade_history)
            
            # Symbol'e göre filtrele
            symbol_trades = df[df['symbol'] == symbol]
            
            if len(symbol_trades) == 0:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'avg_profit': 0,
                    'max_profit': 0,
                    'max_loss': 0,
                    'success_rate': 0
                }
            
            stats = {
                'total_trades': len(symbol_trades),
                'winning_trades': len(symbol_trades[symbol_trades['profit_loss'] > 0]),
                'avg_profit': symbol_trades['profit_loss'].mean(),
                'max_profit': symbol_trades['profit_loss'].max(),
                'max_loss': symbol_trades['profit_loss'].min(),
                'success_rate': (len(symbol_trades[symbol_trades['profit_loss'] > 0]) / len(symbol_trades)) * 100
            }
            
            print(f"\n=== {symbol} İstatistikleri ===")
            print(f"Toplam İşlem: {stats['total_trades']}")
            print(f"Kazanan İşlem: {stats['winning_trades']}")
            print(f"Ortalama Kar: %{stats['avg_profit']:.2f}")
            print(f"En Yüksek Kar: %{stats['max_profit']:.2f}")
            print(f"En Yüksek Zarar: %{stats['max_loss']:.2f}")
            print(f"Başarı Oranı: %{stats['success_rate']:.1f}")
            
            return stats
            
        except Exception as e:
            print(f"İstatistik güncelleme hatası: {str(e)}")
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'avg_profit': 0,
                'max_profit': 0,
                'max_loss': 0,
                'success_rate': 0
            } 

    def analyze_trade_history(self):
        """
        İşlem geçmişini analiz eder ve iyileştirme önerileri sunar
        """
        try:
            df = pd.DataFrame(self.trade_history)
            if len(df) < self.min_trades_for_stats:
                return "Yeterli işlem geçmişi yok"
            
            # Genel istatistikler
            total_trades = len(df)
            winning_trades = len(df[df['profit_loss'] > 0])
            success_rate = (winning_trades / total_trades) * 100
            avg_profit = df[df['profit_loss'] > 0]['profit_loss'].mean()
            avg_loss = df[df['profit_loss'] <= 0]['profit_loss'].mean()
            
            # Zaman bazlı analiz
            df['entry_date'] = pd.to_datetime(df['entry_date'])
            df['hour'] = df['entry_date'].dt.hour
            best_hours = df.groupby('hour')['profit_loss'].mean().sort_values(ascending=False)
            
            # İndikatör bazlı analiz
            rsi_success = df.groupby(pd.cut(df['indicators'].apply(lambda x: x['RSI']), 
                                          bins=[0,30,40,50,60,70,100]))['profit_loss'].mean()
            
            adx_success = df.groupby(pd.cut(df['indicators'].apply(lambda x: x['ADX']), 
                                          bins=[0,20,25,30,35,100]))['profit_loss'].mean()
            
            # İyileştirme önerileri
            recommendations = []
            
            if success_rate < 50:
                recommendations.append("⚠️ Başarı oranı düşük. Sinyal filtrelerini sıkılaştırın.")
            
            if avg_loss < -2.5:
                recommendations.append("🛑 Ortalama kayıp yüksek. Stop-loss seviyelerini daraltın.")
            
            best_hour = best_hours.index[0]
            if best_hours.iloc[0] > best_hours.mean() * 1.5:
                recommendations.append(f"⏰ En karlı işlem saati: {best_hour}:00")
            
            # RSI önerileri
            best_rsi_range = rsi_success.idxmax()
            recommendations.append(f"📊 En başarılı RSI aralığı: {best_rsi_range}")
            
            # ADX önerileri
            best_adx_range = adx_success.idxmax()
            recommendations.append(f"📈 En başarılı ADX aralığı: {best_adx_range}")
            
            # Sonuçları formatla
            analysis = f"""📊 İşlem Analizi ({total_trades} işlem)

💰 Genel Performans:
• Başarı Oranı: %{success_rate:.1f}
• Ortalama Kar: %{avg_profit:.2f}
• Ortalama Zarar: %{avg_loss:.2f}

⚡️ En İyi Performans:
• Saat: {best_hour}:00 (%{best_hours.iloc[0]:.2f})
• RSI: {best_rsi_range} (%{rsi_success.max():.2f})
• ADX: {best_adx_range} (%{adx_success.max():.2f})

🔄 İyileştirme Önerileri:
""" + "\n".join(f"• {r}" for r in recommendations)

            return analysis
            
        except Exception as e:
            print(f"Analiz hatası: {str(e)}")
            return "Analiz yapılamadı"

    def optimize_parameters(self):
        """
        İşlem sonuçlarına göre parametreleri optimize eder
        """
        try:
            df = pd.DataFrame(self.trade_history)
            if len(df) < self.min_trades_for_stats:
                return
            
            # RSI optimizasyonu
            rsi_ranges = df.groupby(pd.cut(df['indicators'].apply(lambda x: x['RSI']), 
                                        bins=[0,30,35,40,45,50,55,60,65,70,100]))['profit_loss'].mean()
            best_rsi_range = rsi_ranges.idxmax()
            
            # ADX optimizasyonu
            adx_ranges = df.groupby(pd.cut(df['indicators'].apply(lambda x: x['ADX']), 
                                        bins=[0,15,20,25,30,35,40,100]))['profit_loss'].mean()
            best_adx_range = adx_ranges.idxmax()
            
            # Stop loss optimizasyonu
            if df['profit_loss'].min() < -3:
                self.stop_loss_percent = 0.015  # Daha sıkı stop-loss
            elif df['profit_loss'].std() > 4:
                self.stop_loss_percent = 0.025  # Daha geniş stop-loss
            
            # Take profit optimizasyonu
            avg_profit = df[df['profit_loss'] > 0]['profit_loss'].mean()
            if avg_profit > 5:
                self.take_profit_percent = avg_profit * 0.8  # Hedefi yükselt
            elif avg_profit < 2:
                self.take_profit_percent = 0.03  # Hedefi düşür
            
            # Sonuçları kaydet
            self.optimized_params = {
                'best_rsi_range': best_rsi_range,
                'best_adx_range': best_adx_range,
                'stop_loss': self.stop_loss_percent,
                'take_profit': self.take_profit_percent,
                'last_update': datetime.now()
            }
            
            print(f"\n=== Parametre Optimizasyonu ===")
            print(f"RSI Aralığı: {best_rsi_range}")
            print(f"ADX Aralığı: {best_adx_range}")
            print(f"Stop Loss: %{self.stop_loss_percent*100:.1f}")
            print(f"Take Profit: %{self.take_profit_percent*100:.1f}")
            
        except Exception as e:
            print(f"Optimizasyon hatası: {str(e)}") 

    def analyze_patterns(self):
        """
        İşlem geçmişindeki başarılı ve başarısız pattern'leri analiz eder
        """
        try:
            df = pd.DataFrame(self.trade_history)
            if len(df) < self.min_trades_for_stats:
                return None
            
            # Pattern analizi
            patterns = {
                'time_patterns': self._analyze_time_patterns(df),
                'indicator_patterns': self._analyze_indicator_patterns(df),
                'price_patterns': self._analyze_price_patterns(df),
                'volume_patterns': self._analyze_volume_patterns(df)
            }
            
            # Başarılı pattern'leri belirle
            successful_patterns = self._identify_successful_patterns(patterns)
            
            return successful_patterns
            
        except Exception as e:
            print(f"Pattern analizi hatası: {str(e)}")
            return None

    def _analyze_time_patterns(self, df):
        """
        Zaman bazlı pattern'leri analiz eder
        """
        df['hour'] = pd.to_datetime(df['entry_date']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['entry_date']).dt.dayofweek
        
        time_patterns = {
            'best_hours': df.groupby('hour')['profit_loss'].mean().sort_values(ascending=False).head(),
            'best_days': df.groupby('day_of_week')['profit_loss'].mean().sort_values(ascending=False),
            'hour_success_rate': df.groupby('hour')['profit_loss'].apply(lambda x: (x > 0).mean() * 100)
        }
        
        return time_patterns

    def _analyze_indicator_patterns(self, df):
        """
        İndikatör bazlı pattern'leri analiz eder
        """
        indicator_patterns = {}
        
        # RSI analizi
        rsi_ranges = pd.cut(df['indicators'].apply(lambda x: x['RSI']), 
                           bins=[0,30,40,50,60,70,100])
        indicator_patterns['rsi_success'] = df.groupby(rsi_ranges)['profit_loss'].agg([
            'mean',
            'count',
            'std',
            lambda x: (x > 0).mean() * 100  # Başarı oranı
        ])
        
        # ADX analizi
        adx_ranges = pd.cut(df['indicators'].apply(lambda x: x['ADX']),
                           bins=[0,20,25,30,40,100])
        indicator_patterns['adx_success'] = df.groupby(adx_ranges)['profit_loss'].agg([
            'mean',
            'count',
            'std',
            lambda x: (x > 0).mean() * 100
        ])
        
        return indicator_patterns

    def _analyze_price_patterns(self, df):
        """
        Fiyat pattern'lerini analiz eder
        """
        price_patterns = {}
        
        # Trend yönü başarı oranı
        price_patterns['trend_success'] = df.groupby(
            df['indicators'].apply(lambda x: x.get('trend', 'Unknown'))
        )['profit_loss'].agg(['mean', 'count', lambda x: (x > 0).mean() * 100])
        
        return price_patterns

    def _analyze_volume_patterns(self, df):
        """
        Hacim pattern'lerini analiz eder
        """
        volume_patterns = {}
        
        # Hacim artış/azalış başarı oranı
        volume_patterns['volume_change_success'] = df.groupby(
            pd.cut(df['indicators'].apply(lambda x: x.get('volume_change', 0)),
                   bins=[-np.inf, -0.5, 0, 0.5, np.inf])
        )['profit_loss'].agg(['mean', 'count', lambda x: (x > 0).mean() * 100])
        
        return volume_patterns

    def _identify_successful_patterns(self, patterns):
        """
        En başarılı pattern'leri belirler
        """
        successful = {
            'time': {
                'best_hours': patterns['time_patterns']['best_hours'].index[
                    patterns['time_patterns']['best_hours'] > 0
                ].tolist(),
                'best_days': patterns['time_patterns']['best_days'].index[
                    patterns['time_patterns']['best_days'] > 0
                ].tolist()
            },
            'indicators': {
                'rsi_ranges': patterns['indicator_patterns']['rsi_success'][
                    patterns['indicator_patterns']['rsi_success']['mean'] > 0
                ].index.tolist(),
                'adx_ranges': patterns['indicator_patterns']['adx_success'][
                    patterns['indicator_patterns']['adx_success']['mean'] > 0
                ].index.tolist()
            },
            'trend': {
                'successful_trends': patterns['price_patterns']['trend_success'][
                    patterns['price_patterns']['trend_success']['mean'] > 0
                ].index.tolist()
            }
        }
        
        return successful 