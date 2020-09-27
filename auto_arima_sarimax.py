import pandas as pd
import numpy as np
from datetime import datetime
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.impute import SimpleImputer

def parser(s):
    return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')

df = pd.read_csv('df_vector_weather_one_hr.csv', parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

df['rain_1h'].replace([np.nan], value = [0], inplace = True)
df['rain_3h'].replace([np.nan], value = [0], inplace = True)
df['snow_1h'].replace([np.nan], value = [0], inplace = True)
df['snow_3h'].replace([np.nan], value = [0], inplace = True)

df['traffic_volume'] = df['traffic_volume'].interpolate(method ='time')
df['speed_avg'] = df['speed_avg'].interpolate(method ='time')
df['seconds_zone'] = df['seconds_zone'].interpolate(method ='time')

df = df['2017-06-01 12:00:00':'2019-08-20 23:30:00']
nobs = int(len(df) * 0.2)
df_train, df_test = df[0:-nobs], df[-nobs:]
train_size = len(df_train)
test_size = len(df_test)

train_y = df_train['traffic_volume']
train_x = df_train[['speed_avg', 'seconds_zone', 'temp', 'pressure',
       'humidity', 'wind_speed', 'wind_deg', 'rain_1h', 'rain_3h', 'snow_1h',
       'snow_3h', 'clouds_all']]

# order = list(map(int,input("\nEnter the oder : ").strip().split()))[:3]
# seasonal_oder = list(map(int,input("\nEnter the seasonal order : ").strip().split()))[:4]
# print("\nSeasonal factor =", seasonal_oder[3])


step_wise=auto_arima(train_y[1:], max_order=None, test='adf', m = 24, seasonal=True,
                      max_P=5, max_D= 1, max_Q=3, 
                     exogenous= train_x[1:],
                      maxiter=40, alpha=0.05, 
                      n_jobs=-1, 
                     information_criterion='aic', 
                      out_of_sample_size=test_size,
                      start_p=2, start_q=1, 
                      max_p=5, max_q=3, 
                      start_d=0, max_d=1,
                      trace=True, 
                      error_action='ignore', 
                      suppress_warnings=True,
                      stepwise=True
                     )
print(step_wise.summary())