from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf;
import pandas as pd
import matplotlib.pyplot as plt

n_samples = 1200   # Number of dataset points
median_income_arr = np.empty(n_samples);
median_house_value_arr = np.empty(n_samples);

steps = 10

cost_arr = np.empty(steps);
step_arr = np.empty(steps);

filename_queue = tf.train.string_input_producer(["housingFiltered.csv"])
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)


record_defaults = [tf.constant([], dtype=tf.float32),tf.constant([], dtype=tf.float32)]
median_income, median_house_value = tf.decode_csv(
    value, record_defaults= record_defaults)
features = tf.stack([median_income, median_house_value])

with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)





  for i in range(n_samples):
    # Retrieve a single instance:
    example, label = sess.run([features, median_income])
    median_income_arr[i] = label;
    #print(median_income_arr[i])
    example, label = sess.run([features, median_house_value])
    median_house_value_arr[i] = label;
    #print(median_house_value_arr[i])

  coord.request_stop()
  coord.join(threads)

plt.subplot(2, 1, 1)
plt.plot(median_income_arr, median_house_value_arr, 'ro')


# start the modelling


# Model linear regression y = Wx + b. x is of shape `num_samples, 1`
x = tf.placeholder(tf.float32, [None, 1])

# Here we initialize W and b to be negative just to illustrate our point!
W = tf.Variable(-100*tf.ones([1, 1]))
b = tf.Variable(-100*tf.ones([1]))

product = tf.matmul(x,W)
y = product + b

# Clipping operation.
clip_W = W.assign(tf.maximum(0., W))
clip_b = b.assign(tf.maximum(0., b))
clip = tf.group(clip_W, clip_b)

y_ = tf.placeholder(tf.float32, [None, 1])

# Cost function
cost = tf.reduce_sum(tf.pow(y-y_, 2))/(2*n_samples)



lr = 0.01

# Training using Gradient Descent to minimize cost
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cost)

init = tf.global_variables_initializer()

median_house_value_arr = median_house_value_arr.reshape(n_samples, 1)
median_income_arr = median_income_arr.reshape(n_samples, 1)

plt.ion()

## SESSION
with tf.Session() as sess:
    sess.run(init)
    for i in range(steps):
        step_arr[i] = i;
        print("*"*40)
        print("Iteration Number %d" %i)
        print("*"*40)
        print("\nBefore gradient computation")
        print("-"*40)

        print("W: %f" % sess.run(W))
        print("b: %f" % sess.run(b))
        feed = { x: median_income_arr, y_: median_house_value_arr }
        sess.run(train_step, feed_dict=feed)

        print("\nAfter gradient computation")
        print("-"*40)
        print("W: %f" % sess.run(W))
        print("b: %f" % sess.run(b))
        print("\nAfter gradient projection")
        print("-"*40)
        # THIS line would ensure the projection step happens!
        sess.run(clip)
        print("W: %f" % sess.run(W))
        print("b: %f" % sess.run(b))
        cost_arr[i] =  sess.run(cost, feed_dict=feed);
        print("*"*40)


        learnt_W = sess.run(W)
        learnt_b = sess.run(b)

        plt.plot(median_income_arr, median_house_value_arr, 'ro')
        pred_Y = np.multiply(median_income_arr, learnt_W)+learnt_b

        plt.subplot(2, 1, 1)
        plt.plot(median_income_arr, pred_Y, 'b')
        plt.title("Y = {:0.2f}X + {:0.2f}".format(learnt_W[0, 0], learnt_b[0]))
        plt.pause(0.05)



plt.subplot(2, 1, 2)
plt.plot(cost_arr, step_arr, 'ro')
plt.title("error")
plt.pause(1000)