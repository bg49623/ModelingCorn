import numpy
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.optimize import curve_fit

pdsiFile = pd.read_csv("Data/MinnesotaMonthlyPDSI.csv")
month = pdsiFile['Date'].astype('float32').to_numpy()
pdsi = pdsiFile['Value'].astype('float32').to_numpy()

N = len(month)

plt.plot(month, pdsi, '.')
plt.show()
