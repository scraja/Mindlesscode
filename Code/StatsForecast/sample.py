from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import pandas as pd


df = pd.read_csv('https://datasets-nixtla.s3.amazonaws.com/air-passengers.csv')
print(df.head())

sf = StatsForecast(models=[AutoARIMA(season_length=12)],freq='M')

sf.fit(df)

forecast_df = sf.predict(h=12,level=[90])
print(forecast_df.tail())

df['ds'] = pd.to_datetime(df['ds'])
sf.plot(df, forecast_df, level=[90])