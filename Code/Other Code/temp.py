import numpy
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.optimize import curve_fit


df = pd.read_csv("juicy.csv")

#getting the mean temperature per day
df['TAVG'] = df[['TMAX','TMIN']].mean(axis = 1)
df['time'] = df.index
temp = df['TAVG'].astype('float32').to_numpy()
time = df['time'].astype('float32').to_numpy()
df = df[pd.notnull(df['time'])]


#turn it into a time series
temp = numpy.nan_to_num(temp)

N = len(time)

def test_func(x,a,b,c,d, e):
    return  a * numpy.sin(b * x + c) + d + e * x

params, _ = curve_fit(test_func, time, temp, p0  = [100, 2 * numpy.pi / 365, 0, -10, 0])

print(params)
plt.plot(time, temp, '.')
plt.plot(time, test_func(time, params[0], params[1], params[2], params[3], params[4]))
plt.show()
