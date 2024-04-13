import streamlit as st
from joblib import load

import requests
from bs4 import BeautifulSoup
from datetime import datetime

st.title('Предсказательная модель')

@st.cache(allow_output_mutation=True)

def load_model():
    model = load('random_forest_model_2017_2023_BTC.joblib')
    return model

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

model = load_model()

def make_prediction(value, day):
    prediction = model.predict([[value, day]])

    return prediction

# Показывам кнопку для запуска
result = st.button('Предскажи')
# Если кнопка нажата, то запускаем
if result:

    inf = get_inf()
    preds = make_prediction(inf.value, inf.day)

    st.write('**Результат предсказания:**')
    st.write('*Модель обучена на данныx 2023 года, предсказание выдает ожидаемый коэфициент умножения (больше 1 - вырасте, меньше - упадет)*')
    st.write(preds)