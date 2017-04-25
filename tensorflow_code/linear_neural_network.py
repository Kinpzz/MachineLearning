import tensorflow as tf
import numpy as np
import pandas as pd
import csv

# add neural layer
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return Weights, biases, outputs
# zero scope normalization
def z_score_normalization(x): 
    mu = np.average(x)
    sigma = np.std(x)
    if sigma == 0:
        x = x * 0
    else :
        x = (x - mu) / sigma; 
    return x;

def MaxMinNormalization(x):
    max = np.max(x)
    min = np.min(x)
    if (max - min) == 0:
        x = 0
    else:
        x = (x - min) / (max - min);  
    return x;

# load data from csv
print "reading data from csv..."
features_read = pd.read_csv('save_train.csv', skiprows=[0],
                usecols=range(1,384), header=None)
label_read = pd.read_csv('save_train.csv', skiprows=[0],
                usecols=[385], header=None)
features = z_score_normalization(np.float32(features_read.values))
label = np.float32(label_read.values)

#tensorflow
print "describe graph"
xs = tf.placeholder(tf.float32,[None, 383])
ys = tf.placeholder(tf.float32,[None, 1])
# add hidden layer
Weights_1, biases_1, l1 = add_layer(xs, 383, 10, activation_function=tf.nn.relu)
# add output layer
Weights_2, biases_2, prediction = add_layer(l1, 10, 1, activation_function=None)

loss = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), 1)))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

# load test data
print "load test data..."
test_x_read = pd.read_csv('save_test.csv', skiprows=[0],
                usecols=range(1,384), header=None)
test_x = z_score_normalization(np.float32(test_x_read.values))
# prediction
xs_t = tf.placeholder(tf.float32,[None, 383])
Wx_plus_b_1 = tf.nn.relu(tf.matmul(xs_t, Weights_1) + biases_1)
test_y = tf.matmul(Wx_plus_b_1, Weights_2) + biases_2

# important step
print "begin training"
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(50000):
        sess.run(train_step, feed_dict={xs: features, ys: label})
        if i % 50 == 0:
            # to see the step improvement
            print(i, sess.run(loss, feed_dict={xs: features, ys: label}))
    predict_y = sess.run(test_y,feed_dict={xs_t: test_x})

# write data
print "wrting data"
with open( './data.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['id','reference'])
    for i in range(25000):
        data = [i] + [predict_y[i][0]]
        print data
        writer.writerow(data)

