from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List
import uvicorn
from datetime import datetime
import asyncio
import threading

from model_trainer import ModelTrainer
from trading_bot import TradingBot
from data_collector import DataCollector
from config import RECOMMENDED_COINS
from trading_signals import SignalGenerator

app = FastAPI(title="Crypto Trading API")

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global değişkenler
active_symbols = set()
trading_bots = {}
latest_signals = {}
signal_lock = threading.Lock()
signal_generators = {}

class TradingSignal(BaseModel):
    symbol: str
    timestamp: datetime
    signal: str
    trend: str
    confidence: float
    price: float
    timeframes: Dict[str, dict]

@app.post("/start_trading/{symbol}")
async def start_trading(symbol: str):
    """
    Tek bir coin için trading başlat
    """
    try:
        # Symbol'ü düzelt (BTCUSDT -> BTC/USDT)
        if "USDT" in symbol:
            formatted_symbol = f"{symbol[:-4]}/USDT"
        else:
            formatted_symbol = f"{symbol}/USDT"
            
        print(f"Trading başlatılıyor: {formatted_symbol}")
        
        if formatted_symbol in active_symbols:
            return {
                "status": "warning",
                "message": f"{formatted_symbol} zaten izleniyor"
            }
            
        # Veri toplayıcı oluştur
        collector = DataCollector()
        historical_data = collector.get_multi_timeframe_data(formatted_symbol)
        
        if historical_data:
            # Trading bot ve sinyal üretici oluştur
            signal_generators[formatted_symbol] = SignalGenerator()
            active_symbols.add(formatted_symbol)
            
            # Sadece 1h timeframe için izleme başlat
            asyncio.create_task(monitor_signals(formatted_symbol, '1h'))
            
            # İlk fiyat bilgisini al
            current_price = collector.get_current_price(formatted_symbol)
            
            return {
                "status": "success",
                "message": f"{formatted_symbol} sinyalleri izleniyor",
                "initial_data": {
                    "price": current_price,
                    "timestamp": datetime.now().isoformat()
                }
            }
        else:
            raise Exception(f"{formatted_symbol} için veri alınamadı")
            
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/stop_trading/{symbol}")
async def stop_trading(symbol: str):
    """
    Coin izlemeyi durdur
    """
    if symbol in active_symbols:
        active_symbols.remove(symbol)
        trading_bots.pop(symbol, None)
        latest_signals.pop(symbol, None)
        return {"message": f"{symbol} trading durduruldu"}
    return {"message": f"{symbol} zaten izlenmiyor"}

@app.get("/signals/{symbol}")
async def get_signals(symbol: str):
    """
    Coin için son sinyalleri getir
    """
    print(f"\n=== Sinyal İsteği Detayları ===")
    print(f"İstenen sembol: {symbol}")
    print(f"Aktif semboller: {active_symbols}")
    print(f"Mevcut sinyaller: {list(latest_signals.keys())}")
    
    try:
        # Önce coinin aktif olup olmadığını kontrol et
        if symbol not in active_symbols:
            print(f"HATA: {symbol} aktif semboller arasında değil!")
            return {
                "status": "error",
                "message": f"{symbol} izlenmiyor",
                "suggestion": "Coini izlemeye almak için /start_multiple_trading endpoint'ini kullanın"
            }
        
        # Sinyal var mı kontrol et
        with signal_lock:
            signal_data = latest_signals.get(symbol)
            print(f"Sinyal verisi: {signal_data}")
            
            if not signal_data:
                print(f"UYARI: {symbol} için henüz sinyal üretilmemiş")
                return {
                    "status": "info",
                    "message": f"{symbol} için henüz sinyal üretilmedi",
                    "timestamp": datetime.now().isoformat(),
                    "is_active": True
                }
            
            return {
                "status": "success",
                "data": signal_data,
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        print(f"HATA: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/active_symbols")
async def get_active_symbols():
    """
    İzlenen coinleri listele
    """
    return {"symbols": list(active_symbols)}

@app.get("/recommended_coins/{category}")
async def get_recommended_coins(category: str = "major"):
    """
    Önerilen coinleri kategori bazında getir
    """
    if category in RECOMMENDED_COINS:
        return {"coins": RECOMMENDED_COINS[category]}
    else:
        available_categories = list(RECOMMENDED_COINS.keys())
        raise HTTPException(
            status_code=400, 
            detail=f"Geçersiz kategori. Mevcut kategoriler: {available_categories}"
        )

@app.post("/start_multiple_trading")
async def start_multiple_trading(category: str = "major", timeframes: str = "1h"):
    """
    Kategori bazında çoklu trading başlat
    """
    try:
        if category not in RECOMMENDED_COINS:
            return {"error": f"Geçersiz kategori. Mevcut kategoriler: {list(RECOMMENDED_COINS.keys())}"}
            
        results = []
        for symbol in RECOMMENDED_COINS[category]:
            result = await start_trading(symbol.replace("/", ""))
            results.append({symbol: result})
            
        return results
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/watch_signals/{symbol}")
async def watch_signals(symbol: str, timeframe: str = "1h"):
    """
    Belirli bir coin için güçlü sinyalleri izlemeye başlar
    """
    try:
        if symbol in active_symbols:
            return {"message": f"{symbol} zaten izleniyor"}
            
        # Veri toplama ve sinyal izleme başlat
        collector = DataCollector([timeframe])
        signal_generator = SignalGenerator()
        
        # Global değişkenlere ekle
        trading_bots[symbol] = signal_generator
        active_symbols.add(symbol)
        
        # Background task olarak sinyal üretmeye başla
        asyncio.create_task(monitor_signals(symbol, timeframe))
        
        return {"message": f"{symbol} {timeframe} sinyalleri izleniyor"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def monitor_signals(symbol, timeframe):
    """
    Belirli bir coin için sinyalleri izler
    """
    print(f"\n=== Monitor Başlatıldı ===")
    print(f"Sembol: {symbol}")
    print(f"Timeframe: {timeframe}")
    
    try:
        if symbol not in signal_generators:
            signal_generators[symbol] = SignalGenerator()
            
        signal_generator = signal_generators[symbol]
        
        while True:
            print(f"\n{symbol} için sinyal kontrolü yapılıyor...")
            
            # Veriyi al
            collector = DataCollector()
            data = collector.get_multi_timeframe_data(symbol)
            
            if data is not None:
                # Timeframe'e göre veriyi al
                df = data.get(timeframe)
                
                if df is not None:
                    try:
                        # Aynı signal generator'ı kullan
                        signal_data = signal_generator.analyze_signals(df, symbol, timeframe)
                        print(f"Sinyal analizi sonucu: {signal_data}")
                        
                        # Son sinyali sakla
                        if signal_data:
                            latest_signals[symbol] = signal_data
                            print(f"Yeni sinyal kaydedildi: {signal_data}")
                            
                    except Exception as e:
                        print(f"Sinyal analiz hatası: {str(e)}")
                else:
                    print(f"HATA: {timeframe} verisi bulunamadı")
            else:
                print(f"HATA: {symbol} için veri alınamadı")
                
            await asyncio.sleep(60)  # 1 dakika bekle
            
    except Exception as e:
        print(f"Monitor hatası ({symbol}): {str(e)}")

@app.post("/stop_all_trading")
async def stop_all_trading():
    """
    Tüm coinlerin izlenmesini durdur
    """
    stopped_symbols = []
    for symbol in list(active_symbols):
        active_symbols.remove(symbol)
        trading_bots.pop(symbol, None)
        signal_generators.pop(symbol, None)  # Signal generator'ı da temizle
        latest_signals.pop(symbol, None)
        stopped_symbols.append(symbol)
    
    return {"message": f"İzleme durduruldu: {stopped_symbols}"}

@app.post("/test_telegram")
async def test_telegram():
    """
    Telegram bildirimlerini test et
    """
    try:
        # Test için geçici bir SignalGenerator oluştur
        test_bot = SignalGenerator()
        
        # Test mesajı gönder
        if test_bot.telegram.send_test_message():
            return {"status": "success", "message": "Test mesajı gönderildi"}
        else:
            raise HTTPException(status_code=500, detail="Telegram mesajı gönderilemedi")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start_all_trading")
async def start_all_trading():
    try:
        all_coins = set()
        for category in RECOMMENDED_COINS.values():
            all_coins.update(category)
            
        started_symbols = []
        trainer = ModelTrainer()  # Tek bir trainer instance'ı
        
        for symbol in all_coins:
            if symbol not in active_symbols:
                collector = DataCollector()
                historical_data = collector.get_multi_timeframe_data(symbol)
                
                if historical_data:
                    # Symbol'ü de parametre olarak geçiyoruz
                    model = trainer.train(historical_data, symbol)
                    trading_bots[symbol] = TradingBot(model)
                    active_symbols.add(symbol)
                    started_symbols.append(symbol)
                    
                    for timeframe in ['15m', '1h', '4h']:
                        asyncio.create_task(monitor_signals(symbol, timeframe))
        
        return {"message": f"Trading başlatıldı: {started_symbols}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_api():
    """
    API'yi başlat
    """
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start_api() 