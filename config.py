# MT5 Configuration
MT5_LOGIN = None  # Demo hesap numarası (opsiyonel)
MT5_PASSWORD = None  # Password (opsiyonel)
MT5_SERVER = None  # Server adı (opsiyonel)

# Strategy Parameters
PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
TIMEFRAME = "M1"  # 1 minute
TRADING_HOURS = (12, 19)  # Budapest 13:00-20:00 = UTC 12:00-19:00
ANALYSIS_HOURS = (11, 12)  # Budapest 12:00-13:00 = UTC 11:00-12:00

# Risk Management
RISK_PER_TRADE = 0.0001  # 0.01% per trade
EMERGENCY_SL_PERCENT = 0.02  # 2%
INDIVIDUAL_TP_PIPS = 5  # TP for each position (scalp)

# News Filter
ENABLE_NEWS_FILTER = True  # Enable/disable news filter
NEWS_BEFORE_MINUTES = 30  # Stop trading X minutes before news
NEWS_AFTER_MINUTES = 30  # Resume trading X minutes after news

# Backtest (SON 30 GÜN - MT5 veri limiti için)
from datetime import datetime, timedelta

end_dt = datetime.now()
start_dt = end_dt - timedelta(days=30)

START_DATE = start_dt.strftime("%Y-%m-%d")
END_DATE = end_dt.strftime("%Y-%m-%d")
INITIAL_BALANCE = 10000

print(f"Backtest period: {START_DATE} to {END_DATE}")

# Advanced Strategy Parameters
MIN_MITIGATION_DISTANCE_PIPS = 0.1  # Mitigation update için min mesafe (scalp)
SWING_LOOKBACK = 2  # Left/Right candles for swing high/low detection

# Volatility Filter
ENABLE_VOLATILITY_FILTER = True  # Enable/disable volatility filter
ATR_PERIOD = 14  # ATR calculation period
ATR_MULTIPLIER = 2.0  # High volatility threshold (current ATR > multiplier * avg ATR)