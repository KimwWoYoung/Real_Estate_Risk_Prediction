import pandas as pd
import numpy as np
import time

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

class RealEstateForecast:

    def __init__(self, data_path):
        self.data_path = data_path
        self.cleaned_data = self.load_data()

    def load_data(self):
        return pd.read_csv(self.data_path)

    @staticmethod
    def create_dataset(dataset, look_back=3):
        dataX, dataY = [], []
        if isinstance(dataset, pd.Series):
            dataset = dataset.values
        for i in range(len(dataset) - look_back - 1):
            dataX.append(dataset[i:(i + look_back)])
            dataY.append(dataset[i + look_back])
        return np.array(dataX), np.array(dataY)

    def train_predict_lstm(self, data_series, epochs=10, batch_size=1):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data_series = scaler.fit_transform(data_series.values.reshape(-1, 1))

        look_back = 3
        trainX, trainY = self.create_dataset(scaled_data_series, look_back)
        trainX = np.reshape(trainX, (trainX.shape[0], look_back, 1))

        model = Sequential()
        model.add(LSTM(100, input_shape=(look_back, 1), return_sequences=True))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(25))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')

        start_time = time.time()
        model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=0)
        end_time = time.time()

        # 모델 저장
        model.save('my_model.h5')

        print(f"모델 학습 시간: {end_time - start_time:.2f}초")

        future_predictions = []
        input_data = trainX[-1]
        for _ in range(3):
            prediction = model.predict(input_data.reshape(1, look_back, 1))
            future_predictions.append(prediction[0, 0])
            input_data = np.roll(input_data, -1)
            input_data[-1] = prediction

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        return future_predictions.flatten()

    @staticmethod
    def convert_to_date_format(date_int):
        return f"{str(date_int)[:4]}-{str(date_int)[4:]}"

    def forecast_and_save(self, city_name):
        city_data = self.cleaned_data[self.cleaned_data['시군구-번지'] == city_name]

        data_city = city_data[['계약년월', '면적당보증금', '면적당매매금']].groupby('계약년월').mean().reset_index()
        data_city['계약년월'] = data_city['계약년월'].apply(self.convert_to_date_format)

        forecasted_매매금 = self.train_predict_lstm(data_city['면적당매매금'])
        forecasted_보증금 = self.train_predict_lstm(data_city['면적당보증금'])

        results = []

        def custom_round(value, decimals=2):
            factor = 10 ** decimals
            rounded_value = round(value * factor) / factor
            return format(rounded_value, f".{decimals}f")

        for month in range(3):
            전세가율 = forecasted_보증금[month] / forecasted_매매금[month]
            rate = custom_round(전세가율, 2)

            if 전세가율 >= 0.9:
                message = "역전세 위험이 매우 높습니다"
            elif 전세가율 >= 0.8:
                message = "역전세 가능성이 높으니 경계해야 합니다"
            elif 전세가율 >= 0.7:
                message = "역전세 가능성에 관심을 기울여야 합니다"
            elif 전세가율 >= 0.6:
                message = "역전세 위험에서 안전합니다"
            else:
                message = "역전세 위험에서 매우 안전합니다"

            results.append({
                "forecast": f"{month+1}개월 후 {city_name}은(는) {message}. 예상 전세가율은 {rate if rate else ''}%",
                "보증금": forecasted_보증금[month],
                "매매금": forecasted_매매금[month]
            })

        return results

    def evaluate_performance(self, true_values, predicted_values):
        mse = mean_squared_error(true_values, predicted_values)
        mae = mean_absolute_error(true_values, predicted_values)
        return mse, mae

def get_city_bunji():
    return input("시군구-번지를 입력하세요: ")

def display_forecast_results(city_bunji, results):
    for result in results:
        print(result['forecast'])
        print(f"매매금 예측값: {result['매매금']}")
        print(f"보증금 예측값: {result['보증금']}")
        print("------------------------------")

if __name__ == '__main__':
    city_bunji = get_city_bunji()
    
    model_instance = RealEstateForecast(data_path=r"C:\Users\rladn\Desktop\Real_Estate_Risk_Prediction\cleaned_data.csv") 
    forecast_results = model_instance.forecast_and_save(city_bunji)

    display_forecast_results(city_bunji, forecast_results)