import ccxt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. تنظیمات اولیه
exchange = ccxt.binance({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET_KEY',
    'enableRateLimit': True
})

symbol = 'BTC/USDT'
timeframe = '4h'

# 2. دریافت داده‌های تاریخی
def fetch_historical_data(symbol, timeframe, limit=1000):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

data = fetch_historical_data(symbol, timeframe)

# 3. مهندسی ویژگی‌های Price Action
def calculate_features(df):
    # ایجاد ویژگی‌های مبتنی بر Price Action
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['body'] = abs(df['close'] - df['open'])
    df['candle_type'] = np.where(df['close'] > df['open'], 1, -1)
    
    # شناسایی الگوهای شمعی
    df['doji'] = np.where(df['body'] / (df['high'] - df['low']) < 0.1, 1, 0)
    df['engulfing'] = np.where(
        (df['candle_type'] == 1) & 
        (df['close'] > df['prev_high']) & 
        (df['open'] < df['prev_low']), 1, 0)
    
    return df.dropna()

data = calculate_features(data)

# 4. آماده‌سازی داده‌ها برای ML
X = data[['prev_high', 'prev_low', 'body', 'candle_type', 'doji', 'engulfing']]
y = np.where(data['close'].shift(-1) > data['close'], 1, 0)  # 1 برای خرید، 0 برای فروش

# حذف آخرین رکورد به دلیل نداشتن برچسب
X = X[:-1]
y = y[:-1]

# 5. آموزش مدل
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 6. ارزیابی مدل
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")

# 7. استراتژی ترید
class TradingStrategy:
    def __init__(self, model):
        self.model = model
        self.position = None
    
    def generate_signal(self, features):
        proba = self.model.predict_proba([features])[0]
        if proba[1] > 0.6:  # آستانه اطمینان
            return 'buy'
        elif proba[0] > 0.6:
            return 'sell'
        else:
            return 'hold'

# 8. اجرای زنده
strategy = TradingStrategy(model)
latest_data = fetch_historical_data(symbol, timeframe, limit=100)
latest_features = calculate_features(latest_data).iloc[-1][['prev_high', 'prev_low', 'body', 'candle_type', 'doji', 'engulfing']]

signal = strategy.generate_signal(latest_features)
print(f"Current Signal: {signal}")

# 9. مدیریت ریسک
def calculate_position_size(balance, risk_per_trade=0.01):
    return balance * risk_per_trade

# 10. اجرای معامله (نمونه ساده)
if signal == 'buy':
    balance = exchange.fetch_balance()['USDT']['free']
    position_size = calculate_position_size(balance)
    exchange.create_market_buy_order(symbol, position_size)
elif signal == 'sell':
    balance = exchange.fetch_balance()[symbol.split('/')[0]]['free']
    exchange.create_market_sell_order(symbol, balance)