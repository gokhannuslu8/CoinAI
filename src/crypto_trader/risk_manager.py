class RiskManager:
    def __init__(self):
        self.max_risk_per_trade = 0.02  # Hesap bakiyesinin maksimum %2'si
        self.max_total_risk = 0.06  # Toplam maksimum %6 risk
        self.max_trades_per_day = 5
        self.min_risk_reward_ratio = 2  # Minimum 1:2 risk/ödül oranı
        
    def calculate_position_size(self, account_balance, entry_price, stop_loss):
        """
        Risk bazlı pozisyon büyüklüğü hesaplama
        """
        risk_amount = account_balance * self.max_risk_per_trade
        price_difference = abs(entry_price - stop_loss)
        position_size = risk_amount / price_difference
        return position_size
        
    def validate_trade(self, entry_price, stop_loss, take_profit):
        """
        Trade'in risk parametrelerine uygunluğunu kontrol eder
        """
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        risk_reward_ratio = reward / risk
        
        return risk_reward_ratio >= self.min_risk_reward_ratio 