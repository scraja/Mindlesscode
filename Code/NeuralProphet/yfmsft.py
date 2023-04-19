import yfinance as yf
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet

msft = yf.Ticker('MSFT')
hist = msft.history(period='max', interval='1d', auto_adjust=True, back_adjust=True)
closeds = pd.DataFrame(columns=["ds","y"])

for key, value in hist['Close'].items():
    closeds.loc[-1] = [key.strftime('%m/%d/%Y'),value]
    closeds.index = closeds.index+1
    closeds.sort_index()



m = NeuralProphet()
mfit = m.fit(closeds,freq='D', epochs=150)
mfuture = m.make_future_dataframe(closeds,periods=730)
mforecast = m.predict(mfuture)
m.plot(mforecast)
m.plot_components(mforecast)
