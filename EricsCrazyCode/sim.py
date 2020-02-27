import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.stats import norm
import trainedmodel


# simulates monthly conditions
def simulate_precipitation():
    sdev = 0.9568056468878128  # determined from data
    mean_increase = .013 / 12  # annual increase/ 12 months
    mean = 1.4838793 + 15 * 12 * mean_increase  # mean from last 30 years
    central_factor = .03

    # seed with last value seen
    forecasts = [6.15]

    for i in range(1, 360):
        last = forecasts[-1]
        step = np.random.normal(0, sdev) + central_factor * (mean - last)
        forecasts.append(last + step)
        mean += mean_increase

    forecasts = np.asarray(forecasts)

    assert len(forecasts) == 360

    return forecasts


def get_temp(t):
    t += 30 * 12
    temp_v = 30.2 * math.sin(0.5236 * t + 4.648) + 0.0019 * t + 38.98
    temp_v += np.random.normal(0, 5)

    return temp_v


def forecast_weather():
    temp = []
    for i in range(0, 360):
        temp.append(get_temp(i))

    temp = np.asarray(temp)

    precipitation = simulate_precipitation()

    assert (len(temp) == len(precipitation) == 360)

    # aggregate monthly results into annual results

    annual_temp = [0] * 30
    annual_precip = [0] * 30

    for i in range(0, 360):
        annual_temp[i // 12] += temp[i]
        annual_precip[i // 12] += precipitation[i]

    for i in range(0, 30):
        annual_temp[i] /= 12
        annual_precip[i] /= 12

    annual_temp = np.asarray(annual_temp)
    annual_precip = np.asarray(annual_precip)

    return annual_temp, annual_precip


losses = []

for trial in range(0, 1000):
    temp, precipitation = forecast_weather()
    loss = trainedmodel.get_prediction(precipitation, temp)
    losses.append(loss)

losses = np.asarray(losses)

plt.hist(losses, weights=[1.0 / len(losses)] * len(losses), bins=50)
plt.title('Relative Frequency of Predicted Losses of Random Walks')
plt.ylabel('Frequency')
plt.xlabel('Predicted Loss Over Next 30 Years in Dollars')
plt.show()

print(losses.mean())
print(losses.std())
print(losses.min())
print(losses.max())
#
# temp, precipitation = forecast_weather()
# loss = trainedmodel.get_prediction(precipitation, temp, True)
