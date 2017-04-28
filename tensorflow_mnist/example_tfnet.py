
"""
    
    Author: Antonio Mejias Gil (anmegi.95@gmail.com)
    Date: Late 2016
    Description: 
        This file creates a convolutional neural network (LeNet-5)
        and trains it on the MNIST dataset.

"""

# Standard modules
import os

# Third-party modules
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Own modules
import settings as st


def create_model_dir(name):
    model_path = os.path.join(st.MODEL_DIR, name)
    os.mkdir(model_path)
    return model_path


# TFNet network
#
# Input: 				            28x28x1
# CONV [n = 32, lxl = 5x5, s = 1]: 	28x28x32
# RELU:                             28x28x32
# POOL [max, lxl = 2x2, s=2]:       14x14x32
# CONV [n = 32, lxl = 5x5, s = 1]:  14x14x64
# RELU:                             14x14x64
# POOL [max, lxl = 2x2, s=2]:       7x7x64
# FC:                               1024
# FC:                               10

# structure initialisation
x = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

# Convolutional, 28x28x1 -> 28x28x32
W_conv1 = tf.Variable(tf.random_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

# Pooling, 28x28x32 -> 14x14x32
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Convolutional, 14x14x32 -> 14x14x64
W_conv2 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

# Pooling, 14x14x64 -> 7x7x64
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Fully connected + dropout, 1024 units
W_fc1 = tf.Variable(tf.random_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully connected, 10 units
W_fc2 = tf.Variable(tf.random_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Loss and training
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_), name='cross_entropy')
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

saver = tf.train.Saver()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.Session()

with sess.as_default():
    sess.run(tf.global_variables_initializer())

    # training for a fixed number of iterations
    for i in range(st.N_STEPS):
        batch = mnist.train.next_batch(st.BATCH_SIZE)

        if i % st.PRINT_FREQ == 0:
            train_acc = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step {}, training accuracy {:.2%}".format(i, train_acc))

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy: {:.2%}".format(
        accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))

    ans = input('Enter model name (leave blank to discard model): ')

    if ans is not '':
        saver.save(sess, create_model_dir(ans) + '/model')
        print('Model saved successfully')

