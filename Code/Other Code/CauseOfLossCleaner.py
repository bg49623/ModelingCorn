import numpy
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import math
import os

# things we kinda care about
# state = 2
# commodity name = 7

# year = 0
# month = 14

# county name = 5

# other misc things:
# 12/13 #16-22, #29-30
os.chdir('Data')

importantColumns = [0, 13, 11, 12, *range(16, 22), 28, 29]

columnText = ['Year', 'Month', 'Cause of Loss Code', 'Cause of Loss Description', 'Policies Earning Premium',
              'Policies Indemnified', 'Net Planted Acres', 'Net Endorsed Acres', 'Liability', 'Total Premium',
              'Indemnity Amount', 'Loss Ratio']
# ^so many brain cells lost

print(importantColumns)
print(columnText)

all_df = pd.DataFrame()

for i in range(1989, 2021):
    df = pd.read_csv('year' + str(i) + '.csv', sep='|', header=None)
    df = df.applymap(lambda x: x.strip() if type(x) == str else x)

    df = df.loc[df[2] == 'MN']
    df = df.loc[df[6] == 'CORN']

    df = df[importantColumns]

    print(i)
    print(df.shape)

    all_df = all_df.append(df, ignore_index=True)

all_df.columns = columnText
print(all_df)

all_df.to_csv('cleanedLoss.txt', index=False)
