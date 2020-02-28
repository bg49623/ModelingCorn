import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.stats import norm


def years_from_1990(year, month):
    # print(month // 12)
    return year - 1992 + month // 12


# indemnity stuff

df = pd.read_csv("Data/cleanedLoss.txt", usecols=['Year', 'Month', 'Indemnity Amount'])

df['Time'] = years_from_1990(df['Year'], df['Month'])
df = df.loc[df['Time'] >= 0]  # drop all rows before time we have loss data for

df = df.drop(columns=['Year', 'Month'])

df = df.groupby(['Time']).sum()

# fill in missing rows, hard coded time steps
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

print('pMean:', precipitation.mean())

# temperature stuff

df3 = pd.read_csv('Data/MinnesotaMonthlyTemperature.csv')
df3['Time'] = years_from_1990(df3['Date'] // 100, (df3['Date'] % 100) + 12)
df3 = df3.groupby(['Time'], as_index=False).mean()

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
    return (x - lower_bound) ** 2


def temp_extremity(x, ideal):
    # y = np.where(x < ideal, ideal - x, 0)
    # return y ** 2
    return (x - ideal) ** 2


def f_predicted_loss(time, ideal_precipitation, ideal_temperature,
                     precipitation_weight,
                     temperature_weight):
    return precipitation_weight * precipitation_extremity(precipitation(time), ideal_precipitation) \
           + temperature_weight * temp_extremity(temperature(time), ideal_temperature)


# scale variables to aid learning

loss_scale = np.linalg.norm(loss)
precipitation_scale = np.linalg.norm(precipitation)
temperature_scale = np.linalg.norm(temperature)

loss = loss / loss_scale
precipitation = precipitation / precipitation_scale
temperature = temperature = temperature / temperature_scale

print(loss_scale)
print(precipitation_scale)
print(temperature_scale)

# save mean temperature for use with initial values

meanTemp = temperature.mean()

# interpolation to make continuous to allow scipy optimize

precipitation = interpolate.interp1d(time_points, precipitation)
temperature = interpolate.interp1d(time_points, temperature)

plt.plot(time_points, precipitation(time_points), label='precip')
plt.plot(time_points, temperature(time_points), label='temp')
plt.plot(time_points, loss, label='loss')
plt.legend()
# plt.show()
# exit()

# plt.plot(time, temperature(time))
# plt.show()

# print(meanTemp)
# params = [0, meanTemp, 1, 1]
params, cov = curve_fit(f_predicted_loss, time_points, loss, p0=[0.19873167, -0.05463327, 2.23379107, 0.67416837],
                        maxfev=10000)

# print(f_predicted_loss(time_points[0], params[0], params[1], params[2], params[3]))

print('params:', params)
print(cov)

param_deviation = np.sqrt(np.diagonal(cov))

param_confidence = param_deviation * norm.ppf(.9)

print(param_deviation)
print(param_confidence)

print(param_confidence[0] * precipitation_scale)
print(param_confidence[1] * temperature_scale)

print()

# plt.plot(time_points, params[2] * precipitation_extremity(precipitation(time_points), params[0])
#          label='precip')
# plt.plot(time_points, params[3] * temp_extremity(temperature(time_points), params[1]), label='temp')

plt.figure(2)

plt.plot(time_points, loss * loss_scale, label='loss')

plt.plot(time_points, f_predicted_loss(time_points, *params) * loss_scale,
         label='predict')
plt.title('Predicted Loss vs. Actual Loss')
plt.xlabel('Years from 1990')
plt.ylabel('Predicted and Actual Loss in Dollars')

bottom, top = plt.ylim()
# plt.ylim(0, top * loss_scale)
plt.legend()
# plt.show()


plt.show()
