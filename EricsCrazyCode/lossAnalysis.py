import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.optimize import curve_fit
from scipy import interpolate


def months_from_1989(year, month):
    return (year - 1989) * 12 + month


# indemnity stuff

df = pd.read_csv("Data/cleanedLoss.txt", usecols=['Year', 'Month', 'Indemnity Amount'])

df['Time'] = months_from_1989(df['Year'], df['Month'])
df = df.drop(columns=['Year', 'Month'])

df = df.groupby(['Time']).sum()

# fill in missing rows, hard coded 373 time steps
df = df.reindex(pd.RangeIndex(373 + 1)).ffill()

time_points = df.index.astype('int').to_numpy()
loss = df['Indemnity Amount'].astype('float32').to_numpy()


# precipitation stuff, read exactly the same as temperature so

def get_values(file_name):
    df2 = pd.read_csv(file_name)

    df2['Time'] = months_from_1989(df2['Date'] // 100, (df2['Date'] % 100))

    df2 = df2.loc[df2['Time'] >= 0]  # drop all rows before time we have loss data for
    df2 = df2.set_index('Time')
    return df2['Value'].astype('float32').to_numpy()


precipitation = get_values('Data/MinnesotaMonthlyPDSI.csv')
temperature = get_values('Data/MinnesotaMonthlyTemperature.csv')

assert (len(time_points) == len(loss) == len(precipitation) == len(temperature))


# we have:
# time, loss, precipitation, temperature


# vector norm
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def extremity(x, ideal):
    return (x - ideal) ** 1


def f_predicted_loss(time, ideal_precipitation, ideal_temperature, precipitation_weight, temperature_weight):
    return precipitation_weight * extremity(precipitation(time), ideal_precipitation) \
           + temperature_weight * extremity(temperature(time), ideal_temperature)


loss = normalize(loss)
precipitation = normalize(precipitation)
temperature = normalize(temperature)

meanTemp = temperature.mean()

precipitation = interpolate.interp1d(time_points, precipitation)
temperature = interpolate.interp1d(time_points, temperature)

# plt.plot(time, precipitation(time))
# plt.plot(time, temperature(time))
# plt.show()

# params = [0, temperature.mean(), 10, 0]
params, _ = curve_fit(f_predicted_loss, time_points, loss, p0=[0, meanTemp, 1, 0])

print(f_predicted_loss(time_points[0], params[0], params[1], params[2], params[3]))

print(params)

plt.plot(time_points, loss)

plt.plot(time_points, f_predicted_loss(time_points, params[0], params[1], params[2], params[3]))
plt.show()
