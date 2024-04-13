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

def make_prediction(value, day):
    prediction = model.predict([[value, day]])

    return prediction

import requests
from bs4 import BeautifulSoup
from datetime import datetime

class inf_Param:
    def __init__(self, value, day):
        self.value = value
        self.day = day

def get_inf():
    url = "https://coinmarketcap.com/currencies/bitcoin/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, features="html.parser")
    elements = soup.find_all('div', class_='sc-f70bb44c-0 sc-cd4f73ae-0 iowNqu flexBetween')

    volume_element = elements[1].find('dd', class_='sc-f70bb44c-0 bCgkcs base-text')
    volume_str = volume_element.text
    volume_str = volume_str.split('$')
    volume_str = ''.join(filter(str.isdigit, volume_str[-1]))
    volume_int = int(volume_str)  # объем продаж

    priceBTC_element = soup.find_all('div', class_='sc-aef7b723-0 sc-43ae580a-1 dCXUwm')
    price_str = priceBTC_element[1].find('input')['value']
    price_str = float(price_str)
    price_int = int(price_str)# текущая цена

    now = datetime.now()
    weekday = now.weekday()

    ans = inf_Param(volume_int/price_int, weekday)

    return ans

inf = get_inf()
print(inf.value, inf.day)

print(make_prediction(inf.value, inf.day))
