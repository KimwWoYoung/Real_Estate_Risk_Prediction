import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from keras.layers import Dropout

plt.rcParams['font.family'] = 'NanumGothic'

def load_data(file_path):
    return pd.read_csv(file_path)


# 과거의 3개 데이터 포인트를 기반으로 예측하도록 X,Y 모델에 적합하게 맞추기
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

#  LOOK_BACK 지정
look_back = 3

def train_predict_lstm(data_series, epochs=50, batch_size=1):
    # 데이터 스케일링
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_series.values.reshape(-1, 1))

    # X,Y 나누기
    X, y = create_dataset(scaled_data, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        # LSTM model
    model = Sequential()
    model.add(LSTM(100, input_shape=(look_back, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(25))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    # 모델 기반으로 예측하기
    input_data = scaled_data[-look_back:]
    predictions = []

    for _ in range(3):
        pred = model.predict(input_data.reshape(1, look_back, 1))
        predictions.append(pred[0, 0])
        input_data = np.roll(input_data, -1)
        input_data[-1] = pred
    
    # 스케일링된 예측값 스케일링 이전으로 바꾸기 
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.reshape(-1)


def convert_to_date_format(date_int):
    date_str = str(date_int)
    return date_str[:4] + '-' + date_str[4:]



def forecast_and_save(city_name, city_group, folder_path):
    # 시군구-번지로 필터링
    city_data = cleaned_data[cleaned_data['시군구-번지'] == city_name]

    # 계약년월으로 GROUPBY하고 계약년월 문자열로 바꾸고 시계열데이터로 변환하기
    data_city = city_data[['계약년월', '면적당보증금', '면적당매매금','전세가율']].groupby('계약년월').mean().reset_index()
    data_city['계약년월'] = data_city['계약년월'].apply(convert_to_date_format)
    data_city['timestep'] = range(len(data_city))

    # TRAIN 데이터 모델 적용
    forecasted_매매금 = train_predict_lstm(data_city['면적당매매금'])
    forecasted_보증금 = train_predict_lstm(data_city['면적당보증금'])

    # 그래프에 나타낼 예측계약년월 추가하기 
    dates = data_city['계약년월'].tolist() + ['2023-08','2023-09', '2023-10']

    # 그래프
    plt.figure(figsize=(14,7))
    actual_values_매매금 = data_city['면적당매매금'].values
    predicted_values_매매금 = np.concatenate((actual_values_매매금, forecasted_매매금))
    plt.plot(dates, predicted_values_매매금, label='Predicted 면적당매매금', color='blue', linestyle='--')
    plt.plot(dates[:len(actual_values_매매금)], actual_values_매매금, label='Actual 면적당매매금', color='blue')

    # Plot for 면적당보증금
    actual_values_보증금 = data_city['면적당보증금'].values
    predicted_values_보증금 = np.concatenate((actual_values_보증금, forecasted_보증금))
    plt.plot(dates, predicted_values_보증금, label='Predicted 면적당보증금', color='red', linestyle='--')
    plt.plot(dates[:len(actual_values_보증금)], actual_values_보증금, label='Actual 면적당보증금', color='red')


    plt.axvline(x='2023-07', color='black', linestyle='--')
    plt.title(f'면적당매매금 & 면적당보증금 Forecast for {city_group} using LSTM')
    plt.xlabel('계약년월')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # PNG 파일 형식으로 저장하기 
    file_name = os.path.join(folder_path, city_name.replace(" ", "_") + ".png")
    plt.savefig(file_name)
    plt.close()


if __name__ == "__main__":
    data_path = r"C:\Users\rladn\Downloads\cleaned_data.csv"
    cleaned_data = load_data(data_path)
    cleaned_data = cleaned_data[306624:]

    temp_folder = "forecast_pngs"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    
    unique_cities = cleaned_data['시군구-번지'].unique()
    for city in unique_cities:
        forecast_and_save(city, city, temp_folder)

    # ZIP형식으로 PNG 파일 저장하기 
    shutil.make_archive("forecasted_graphs", 'zip', temp_folder)
    

    shutil.rmtree(temp_folder)

"forecasted_graphs.zip file has been created successfully!"
