import pandas as pd

data = pd.read_csv('bitcoin_2017_to_2023.csv')

data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.sort_values('timestamp')
data = data.drop('low', axis=1)
data = data.drop('high', axis=1)
data = data.drop('close', axis=1)
data = data.drop('quote_asset_volume', axis=1)
data = data.drop('number_of_trades', axis=1)
data = data.drop('taker_buy_base_asset_volume', axis=1)
data = data.drop('taker_buy_quote_asset_volume', axis=1)

data.set_index('timestamp', inplace=True)

data['volume_24h'] = data['volume'].rolling(window=1440).sum()
data['volume_24h'].fillna(1, inplace=True)

data = data.resample('H').last()

data['weekday'] = data.index.weekday
data['target'] = data['open'].shift(-1) / data['open']
data.at[data.index[-1], 'target'] = 0

data = data.drop(data.head(24).index)
data = data.drop(data.tail(1).index)

data = data.drop('open', axis=1)

data.set_index('volume', inplace=True)
data.dropna(inplace=True)


from sklearn.model_selection import train_test_split

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train.values, y_train.values)

y_pred = model.predict(X_test.values)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

from joblib import dump

dump(model, 'random_forest_model_2017_2023_BTC.joblib')

