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
async def start_trading(symbol: str, timeframes: str = "15m,1h,4h"):
    """
    Yeni bir coin için trading başlat
    """
    try:
        if symbol in active_symbols:
            return {"message": f"{symbol} zaten izleniyor"}
            
        # Trading başlat
        collector = DataCollector(timeframes=timeframes.split(','))
        historical_data = collector.get_multi_timeframe_data(symbol)
        
        if not historical_data:
            raise HTTPException(status_code=400, detail=f"{symbol} için veri alınamadı")
            
        trainer = ModelTrainer()
        model = trainer.train(historical_data)
        bot = TradingBot(model)
        
        # Global değişkenlere ekle
        trading_bots[symbol] = bot
        active_symbols.add(symbol)
        
        # Background task olarak sinyal üretmeye başla
        asyncio.create_task(generate_signals(symbol, timeframes))
        
        return {"message": f"{symbol} trading başlatıldı"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    if symbol not in active_symbols:
        raise HTTPException(status_code=404, detail=f"{symbol} izlenmiyor")
        
    with signal_lock:
        return latest_signals.get(symbol, {"message": "Henüz sinyal yok"})

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
async def start_multiple_trading(category: str = "major", timeframe: str = "1h"):
    """
    Bir kategorideki tüm coinler için sinyal izlemeyi başlat
    """
    try:
        if category not in RECOMMENDED_COINS:
            raise HTTPException(
                status_code=400,
                detail=f"Geçersiz kategori. Mevcut kategoriler: {list(RECOMMENDED_COINS.keys())}"
            )
        
        # Önce veri toplayıcıyı oluştur
        collector = DataCollector([timeframe])
        results = []
        
        for symbol in RECOMMENDED_COINS[category]:
            try:
                # Önce veriyi kontrol et
                data = collector.get_multi_timeframe_data(symbol)
                if not data or timeframe not in data:
                    results.append({
                        symbol: {
                            "status": "error",
                            "message": f"{symbol} için veri alınamadı"
                        }
                    })
                    continue
                
                # Coin zaten izleniyorsa, durumu bildir
                if symbol in active_symbols:
                    results.append({
                        symbol: {
                            "status": "warning",
                            "message": f"{symbol} zaten izleniyor"
                        }
                    })
                    continue
                
                # Yeni signal generator oluştur
                signal_generator = SignalGenerator()
                
                # Global değişkenlere ekle
                trading_bots[symbol] = signal_generator
                active_symbols.add(symbol)
                
                # Background task olarak izlemeye başla
                asyncio.create_task(monitor_signals(symbol, timeframe))
                
                results.append({
                    symbol: {
                        "status": "success",
                        "message": f"{symbol} {timeframe} sinyalleri izleniyor",
                        "initial_data": {
                            "price": float(data[timeframe]['close'].iloc[-1]),
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                })
                
            except Exception as e:
                results.append({
                    symbol: {
                        "status": "error",
                        "message": str(e)
                    }
                })
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    try:
        # Her sembol için bir SignalGenerator oluştur veya var olanı kullan
        if symbol not in signal_generators:
            signal_generators[symbol] = SignalGenerator()
            
        signal_generator = signal_generators[symbol]
        
        while True:
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
                        
                        # Son sinyali sakla
                        if signal_data:
                            latest_signals[symbol] = signal_data
                            
                    except Exception as e:
                        print(f"Sinyal analiz hatası ({symbol}): {str(e)}")
                        
            await asyncio.sleep(60)  # 1 dakika bekle
            
    except Exception as e:
        print(f"Sinyal izleme hatası ({symbol}): {str(e)}")

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