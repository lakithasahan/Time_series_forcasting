import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
df = pd.read_csv('AAPL.csv')

close_data = df['Close']


def lag_view(x, order):
    """
    For every value X_i create a row that lags k values: [X_i-1, X_i-2, ... X_i-k]
    """
    y = x.copy()
    # Create features by shifting the window of `order` size by one step.
    # This results in a 2D array [[t1, t2, t3], [t2, t3, t4], ... [t_k-2, t_k-1, t_k]]
    x = np.array([y[-(i + order):][:order] for i in range(y.shape[0])])

    # Reverse the array as we started at the end and remove duplicates.
    # Note that we truncate the features [order -1:] and the labels [order]
    # This is the shifting of the features with one time step compared to the labels
    x = np.stack(x)[::-1][order - 1: -1]
    y = y[order:]

    return x, y

order=2
lag = lag_view(close_data, order)[0]

#print(lag.sum(axis=1)/order)
mv_out=lag.sum(axis=1)/order
plt.plot(mv_out,'r')

plt.show()





e = close_data
N=len(close_data)
q =2
b =np.random.normal(0, 0.1, size=q)
b = np.r_[1, b][::-1]
z = np.zeros(N)

for i in range(q, N):
    for j in range(q):
        z[i] = b[j]*e[i-j]
    z[i]=z[i]+e[i]

plt.plot(z[q:],'g')
plt.show()






p=3
c=0
e = z
N=len(z)
a = np.random.normal(0, 0.1, size=p)
x = np.zeros(N)
a = np.r_[1, a][::-1]
for i in range(p, N):
    for j in range(p):
        x[i] = a[j]*x[i-j]
    x[i]=x[i]+c+e[i]


plt.plot(x[p:])
plt.show()




model = ARMA(close_data, order=(2,3))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(0, len(close_data))
plt.plot(yhat[3:],'b')
plt.show()







""""
ar plot
e = mv_out
N=len(mv_out)
a = [0.2, 0.2]
# AR model
#a = [0.5,0.5]
p = len(a)
c=-100
x = np.zeros(N)
for i in range(p, N):
    x[i] = c+a[0]*x[i-2] + a[1]*x[i-1] + e[i]


plt.plot(x)
plt.show()
"""





""""
phi = np.random.normal(0, 0.1, size=100)
plt.hist(phi)
plt.show()

"""


"""
for i in range(0, 11, 5):
    phi = np.random.normal(0, 0.1, size=i + 1)
    phi = np.r_[1, phi][::-1]
    print(phi)
    
"""