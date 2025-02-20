import argparse
from model_trainer import ModelTrainer
from trading_bot import TradingBot
from data_collector import DataCollector

def main():
    parser = argparse.ArgumentParser(description='Kripto Trading Bot')
    parser.add_argument('--symbol', type=str, default='SOLUSDT', help='Trading yapılacak coin (örn: SOLUSDT)')
    parser.add_argument('--timeframes', type=str, default='15m,1h,4h', help='Analiz edilecek zaman dilimleri (virgülle ayrılmış)')
    parser.add_argument('--backtest', action='store_true', help='Backtest modunu aktifleştirir')
    
    args = parser.parse_args()
    
    try:
        # Veri toplama
        collector = DataCollector(timeframes=args.timeframes.split(','))
        historical_data = collector.get_multi_timeframe_data(args.symbol)
        
        if historical_data is None:
            print(f"{args.symbol} için veri toplanamadı")
            return
            
        # Model eğitimi
        trainer = ModelTrainer()
        model = trainer.train(historical_data)
        
        # Trading bot
        bot = TradingBot(model)
        
        # Market bilgilerini al
        market_info = collector.get_market_info(args.symbol)
        
        # Trading kararını al ve göster
        decisions = bot.get_trading_decision(args.symbol, historical_data)
        bot.display_analysis(args.symbol, market_info, decisions, historical_data)
        
    except Exception as e:
        print(f"Hata: {str(e)}")

if __name__ == "__main__":
    main() 