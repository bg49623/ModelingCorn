import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.stats import norm


# forecasting values from 2020 onwards
def years_from_2020(year, month):
    # print(month // 12)
    return year - 2020 + month // 12


# precipitation stuff

df2 = pd.read_csv('Data/MinnesotaMonthlyPDSI.csv')

# model delay removed as it's a random walk anyways
# formatting below is due to the unique nature of date in the temp data
# you can change it to a different format if you wish
df2['Time'] = years_from_2020(df2['Date'] // 100, (df2['Date'] % 100))
df2 = df2.groupby(['Time'], as_index=False).mean()

df2 = df2.loc[df2['Time'] >= 0]  # drop all rows before time we have loss data for
df2 = df2.set_index('Time')
precipitation = df2['Value'].astype('float32').to_numpy()

precipitation = precipitation[:-1]  # remove extra due to added

# temperature stuff

df3 = pd.read_csv('Data/MinnesotaMonthlyTemperature.csv')
df3['Time'] = years_from_2020(df3['Date'] // 100, (df3['Date'] % 100))
df3 = df3.groupby(['Time'], as_index=False).min()
# we use min temperature as that seems to be most important for crop production

df3 = df3.loc[df3['Time'] >= 0]  # drop all rows before time we have loss data for
df3 = df3.set_index('Time')
temperature = df3['Value'].astype('float32').to_numpy()

temperature = temperature[:-1]

assert (len(precipitation) == len(temperature))


# above code is all for data import, edit at will

def precipitation_extremity(x, ideal):
    return (x - ideal) ** 2


def temp_extremity(x, ideal):
    return (x - ideal) ** 2


def f_predicted_loss(time, ideal_precipitation, ideal_temperature,
                     precipitation_weight,
                     temperature_weight):
    return precipitation_weight * precipitation_extremity(precipitation[time], ideal_precipitation) \
           + temperature_weight * temp_extremity(temperature[time], ideal_temperature)


# scale variables to aid learning

loss_scale = 1643741800.0
precipitation_scale = 12.076221
temperature_scale = 52.896217

precipitation = precipitation / precipitation_scale
temperature = temperature = temperature / temperature_scale

# learned parameters
params = [0.19873167, -0.05463327, 2.23379107, 0.67416837]

# set end date

times = np.arange(0, len(precipitation))

predicted_loss = f_predicted_loss(times, *params)

# summary statistics: save them however you want, remember to rescale by loss scale!!!!!!
# over multiple runs just use the average value of total loss
total_loss = predicted_loss.sum() * loss_scale

print(total_loss)

# visualization tools just in case
plt.plot(times, temperature, label='temperature')
plt.plot(times, precipitation, label='precipitation')
plt.plot(times, predicted_loss, label='predicted loss')
plt.legend()
plt.show()
