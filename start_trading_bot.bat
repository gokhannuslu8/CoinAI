@echo off
echo Trading Bot başlatılıyor...
cd /d %~dp0

REM API'yi başlat (arka planda)
start /B python src/crypto_trader/api.py > trading_bot.log 2>&1

REM 5 saniye bekle (API'nin başlaması için)
timeout /t 5

REM Gaming kategorisini başlat
curl -X POST "http://localhost:8000/start_multiple_trading?category=gaming"

REM DeFi kategorisini başlat
curl -X POST "http://localhost:8000/start_multiple_trading?category=defi"

REM Major kategorisini başlat
curl -X POST "http://localhost:8000/start_multiple_trading?category=major"

REM Test mesajı gönder
curl -X POST "http://localhost:8000/test_telegram"

echo.
echo Trading Bot başlatıldı ve coinler izleniyor...
echo Telegram bildirimlerini kontrol edebilirsiniz.
echo Bu pencereyi küçültebilirsiniz.
echo Kapatmak için Ctrl+C kullanın.

REM Pencereyi açık tut
pause > nul 