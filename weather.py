import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random as random
import math

df = pd.read_csv("juicy.csv")
df['TAVG'] = df[['TMAX','TMIN']].mean(axis = 1)
df['time'] = df.index
temp = df['TAVG'].astype('float32')
time = df['time']
df = df[pd.notnull(df['time'])]

#introducing tensors
#t (time) and y (precipitation) for our training data
t = tf.placeholder("float32")
y = tf.placeholder("float32")
temp = np.nan_to_num(temp)

#initializing a,b,c,d for our model (randomly)
random.seed(1)
days_in_a_year = tf.constant(365.25, dtype = "float32")
w = tf.Variable(np.random.sample(4), name = "w", dtype = "float32")
#Deploying our model
model = w[0] + w[1] * t + w[2] * tf.sin(2 * math.pi * (t- w[3])/days_in_a_year)
cost = tf.reduce_mean(tf.square(y - model))

optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

#starting up tf session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(0, 100000):
        sess.run(optimizer, feed_dict={t: time, y: temp})
        if i % 10000 == 0:
            print(sess.run(cost, feed_dict = {t:time, y:temp}))
            print(sess.run(w, feed_dict={t: time, y: temp}))




