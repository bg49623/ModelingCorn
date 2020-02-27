import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.stats import norm

pdsi = pd.read_csv('Data/MinnesotaMonthlyPDSI.csv')
pdsis = np.array(pdsi['Value'].values)

# diffarr = np.diff(pdsis)
# plt.plot(range(0, len(diffarr)), diffarr[range(0, len(diffarr))])
#
# plt.show()
# exit()

sdev = 0.9568056468878128
mean_increase = .013 / 12  # annual increase/ 12 months
mean = 1.4838793 + 15 * 12 * mean_increase  # mean from last 30 years central_factor = .04
central_factor = .03

print(sdev)

# seed with last value seen
forecasts = [6.15]

for i in range(1, 360):
    last = forecasts[-1]
    step = np.random.normal(0, sdev) + central_factor * (mean - last)
    forecasts.append(last + step)
    mean += mean_increase

forecasts = np.asarray(forecasts)

assert len(forecasts) == 360

forecasts = np.asarray(forecasts)

combined = np.concatenate([pdsis, forecasts])

# original contains all values except last
plt.plot(np.arange(0, len(pdsis) - 1), pdsis[:-1], label='Historical Data')

# start from first
int2 = np.arange(len(pdsis), len(combined))
print(int2.shape)

plt.plot(int2, combined[int2], 'r', label='Simulated Data')
plt.ylabel('PDSI')
plt.xlabel('Months from 1900')
plt.title('Historical and Simulated PDSI')
plt.legend()

plt.figure(2)
plt.ylabel('PDSI')
plt.xlabel('Months from 1900')
plt.title('Historical and Simulated PDSI in the Same Color')

plt.plot(range(0, len(combined)), combined[range(0, len(combined))])
plt.show()
