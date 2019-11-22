import itertools
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import statsmodels.api as sm
import numpy as np

matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['text.color'] = 'k'


df = pd.read_csv("AAPL.csv", parse_dates=["Date"], index_col="Date")
print(df)

data_ = df.groupby('Date')['Close'].sum().reset_index()
data_ = data_.set_index('Date')
sampled_data = data_['Close'].resample('w').mean()



# rcParams['figure.figsize'] = 30, 10
plt.xticks(rotation='vertical')
plt.plot(sampled_data)
plt.title('APPLE Inc Stock close Price Prediction')
plt.xlabel('Date')
plt.ylabel('AAPL -Apple Stock prices (Close)')
plt.legend()
plt.show()

rcParams['figure.figsize'] = 10, 8
decomposition = sm.tsa.seasonal_decompose(sampled_data, model='additive')
fig = decomposition.plot()
plt.show()

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print(seasonal_pdq)
print(pdq)

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

"""
parameters=[]
aic_array=[]

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(sampled_data, order=param, seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit(disp=0)
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            parameters.append(str(param)+str(param_seasonal))
            aic_array.append(results.aic)
        except:
            continue


print(parameters)
print(aic_array)

best_parameters=parameters[aic_array.index(min(aic_array))]
print("Best parameters with lowest aic -"+best_parameters)
"""

mod = sm.tsa.statespace.SARIMAX(sampled_data,
                                order=(0, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit(disp=0)
print(results)
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 8))
plt.show()


pred = results.get_prediction(start='2017-12-31', end='2019-09-22', dynamic=False)
pred_ci = pred.conf_int()
print(pred_ci)

ax = sampled_data['2014':].plot(label='Actual Stock Close Plot')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.9, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

plt.title('APPLE Inc Stock close Price Prediction')
ax.set_xlabel('Date')
ax.set_ylabel('AAPL -Apple Stock prices (Close)')
plt.legend()
plt.show()



mse = ((pred.predicted_mean - sampled_data['2014':]) ** 2).mean()
print('The Mean Squared Error is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error is {}'.format(round(np.sqrt(mse), 2)))


data_for_forcast = sampled_data[0:int(len(sampled_data) * 0.9)]
mod = sm.tsa.statespace.SARIMAX(data_for_forcast,
                                order=(0, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit(disp=0)

forcast_pred = results.get_forecast(steps=20)
print(forcast_pred)
pred_ci = forcast_pred.conf_int()
print(pred_ci)


ax = data_for_forcast.plot(label='Actual Stock Close Plot')
forcast_pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.9, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

plt.title('APPLE Inc Stock close Price Prediction')
ax.set_xlabel('Date')
ax.set_ylabel('AAPL -Apple Stock prices (Close)')
plt.legend()
plt.show()
