import pandas as pd
import numpy as np
from datetime import datetime
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.impute import SimpleImputer

def parser(s):
    return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')

df = pd.read_csv('df_vector_weather_one_hr.csv', parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

df = df['2017-06-01 00:00:00':'2019-08-21 00:00:00']

df['rain_1h'].replace([np.nan], value = [0], inplace = True)
df['rain_3h'].replace([np.nan], value = [0], inplace = True)
df['snow_1h'].replace([np.nan], value = [0], inplace = True)
df['snow_3h'].replace([np.nan], value = [0], inplace = True)

df['traffic_volume'] = df['traffic_volume'].interpolate(method ='time')
df['speed_avg'] = df['speed_avg'].interpolate(method ='time')
df['seconds_zone'] = df['seconds_zone'].interpolate(method ='time')

nobs = int(len(df) * 0.2)
df_train, df_test = df[0:-nobs], df[-nobs:]
train_size = len(df_train)
review_size = len(df_train)
test_size = len(df_test)

from statsmodels.tsa.stattools import acf
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    mse = np.mean((forecast - actual)**2) #mse
    return({'mape':mape, 'me':me, 'mae': mae, 'mse' : mse, 'rmse':rmse})

train_y = df_train['traffic_volume']
train_x = df_train[['speed_avg', 'seconds_zone', 'temp', 'pressure',
       'humidity', 'wind_speed', 'wind_deg', 'rain_1h', 'rain_3h', 'snow_1h',
       'snow_3h', 'clouds_all']]

test_y = df_test['traffic_volume']
test_x = df_test[['speed_avg', 'seconds_zone', 'temp', 'pressure',
       'humidity', 'wind_speed', 'wind_deg', 'rain_1h', 'rain_3h', 'snow_1h',
       'snow_3h', 'clouds_all']]

model= SARIMAX(train_y, exog=train_x, order=(2,0,2), seasonal_order=(3,0,3,24))
results= model.fit()

fc = results.forecast(test_size, alpha=0.05, exog=test_x)  # 95% conf
accuracy_prod = forecast_accuracy(fc, df_test['traffic_volume'].values)
print(accuracy_prod)

results.save('SARIMAX_data_vector_congress_ave_season.pkl')
