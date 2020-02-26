import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.optimize import curve_fit
from scipy import interpolate


def years_from_1990(year, month):
    # print(month // 12)
    return year - 1992 + month // 12


# indemnity stuff

df = pd.read_csv("Data/cleanedLoss.txt", usecols=['Year', 'Month', 'Indemnity Amount'])

df['Time'] = years_from_1990(df['Year'], df['Month'])
df = df.loc[df['Time'] >= 0]  # drop all rows before time we have loss data for

df = df.drop(columns=['Year', 'Month'])

df = df.groupby(['Time']).sum()

# fill in missing rows, hard coded 373 time steps
df = df.reindex(pd.RangeIndex(29)).ffill()

time_points = df.index.astype('int').to_numpy()
loss = df['Indemnity Amount'].astype('float32').to_numpy()

# precipitation stuff

df2 = pd.read_csv('Data/MinnesotaMonthlyPDSI.csv')

# add 12 months to model delay as the impact of drought/excess water seems to take a year to take effect
df2['Time'] = years_from_1990(df2['Date'] // 100, (df2['Date'] % 100) + 12)
df2 = df2.groupby(['Time'], as_index=False).mean()

df2 = df2.loc[df2['Time'] >= 0]  # drop all rows before time we have loss data for
df2 = df2.set_index('Time')
precipitation = df2['Value'].astype('float32').to_numpy()

precipitation = precipitation[:-1]  # remove extra due to added

# temperature stuff

df3 = pd.read_csv('Data/MinnesotaMonthlyTemperature.csv')
df3['Time'] = years_from_1990(df3['Date'] // 100, (df3['Date'] % 100) + 12)
df3 = df3.groupby(['Time'], as_index=False).min()

df3 = df3.loc[df3['Time'] >= 0]  # drop all rows before time we have loss data for
df3 = df3.set_index('Time')
temperature = df3['Value'].astype('float32').to_numpy()

temperature = temperature[:-1]

assert (len(time_points) == len(loss) == len(precipitation) == len(temperature))


# we have:
# time, loss, precipitation, temperature


# vector norm
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def precipitation_extremity(x, lower_bound):
    # y = np.where(x < lower_bound, lower_bound - x, 0)
    # z = np.where(x > upper_bound, x - upper_bound, 0)
    # return y ** 2 + z ** 2
    return abs(x - lower_bound) ** 3


def temp_extremity(x, ideal):
    y = np.where(x < ideal, ideal - x, 0)
    # return y ** 2
    return (x - ideal) ** 2


def f_predicted_loss(time, ideal_precipitation, ideal_temperature,
                     precipitation_weight,
                     temperature_weight):
    return precipitation_weight * precipitation_extremity(precipitation(time), ideal_precipitation) \
           + temperature_weight * temp_extremity(temperature(time), ideal_temperature)


scale_factor = np.linalg.norm(loss)
loss = normalize(loss)
precipitation = normalize(precipitation)
temperature = normalize(temperature)

meanTemp = temperature.mean()

precipitation = interpolate.interp1d(time_points, precipitation)
temperature = interpolate.interp1d(time_points, temperature)

plt.plot(time_points, precipitation(time_points), label='precip')
plt.plot(time_points, temperature(time_points), label='temp')
plt.plot(time_points, loss, label='loss')
plt.legend()
plt.show()
# exit()

# plt.plot(time, temperature(time))
# plt.show()

params = [0, meanTemp, 1, 1]
params, _ = curve_fit(f_predicted_loss, time_points, loss, p0=[-.1, meanTemp, 1, 1])

print(f_predicted_loss(time_points[0], params[0], params[1], params[2], params[3]))

print(params)

# plt.plot(time_points, params[3] * precipitation_extremity(precipitation(time_points), params[0], params[1]),
#          label='precip')
# plt.plot(time_points, params[4] * temp_extremity(temperature(time_points), params[2]), label='temp')

plt.figure(2)

plt.plot(time_points, loss, label='loss')

plt.plot(time_points, f_predicted_loss(time_points, params[0], params[1], params[2], params[3]),
         label='predict')
plt.legend()
# plt.show()

plt.show()
