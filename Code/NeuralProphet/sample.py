from neuralprophet import NeuralProphet
import pandas as pd

data_location = "https://raw.githubusercontent.com/ourownstory/neuralprophet-data/main/datasets/"
df = pd.read_csv(data_location + "wp_log_peyton_manning.csv")

print(df.tail())

m = NeuralProphet()
metrics = m.fit(df)

forecast = m.predict(df)

forecasts_plot = m.plot(forecast)

