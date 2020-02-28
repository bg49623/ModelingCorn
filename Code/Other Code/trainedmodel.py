import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.stats import norm

# we need precipitation and temperature values
#

precipitation = np.asarray([])
temperature = np.asarray([])


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


def get_prediction(p_input, t_input, flag=False):
    global precipitation
    global temperature

    precipitation = p_input
    temperature = t_input
    assert (len(precipitation) == len(temperature))

    # scale variables to aid learning

    loss_scale = 1643741800.0
    precipitation_scale = 12.076221
    temperature_scale = 224.38762

    precipitation = precipitation / precipitation_scale
    temperature = temperature = temperature / temperature_scale

    # learned parameters
    params = [1.99557291e-01, -2.61180649e+02, 2.54541569e+00, 3.69657729e-07]

    # set end date

    times = np.arange(0, len(precipitation))

    predicted_loss = f_predicted_loss(times, *params)

    predicted_loss *= loss_scale

    # summary statistics: save them however you want, remember to rescale by loss scale!!!!!!
    # over multiple runs just use the average value of total loss
    total_loss = predicted_loss.sum()

    if flag:
        # plt.plot(times, temperature, label='temperature')
        # plt.plot(times, precipitation, label='precipitation')
        plt.plot(times, predicted_loss[times])

        plt.title('Predicted Loss')
        plt.xlabel('Years from 2020')
        plt.ylabel('Predicted Loss in Dollars')

        # bottom, top = plt.ylim()
        # plt.ylim(0, top * loss_scale)

        # plt.legend()
        plt.show()

    return total_loss
