
"""

    Author: Antonio Mejias Gil (anmegi.95@gmail.com)
    Date: Late 2016
    Description:
        This file shows the very basic functionality of TensorFlow applied to the
        classic Iris flower classification problem. It allows the user to train
        a model, save it and obverve various learning curves.
        The graph has been written so as to resemble the notation and approach in
        the FDP report (for instance, inputs and targets as columns).

"""

# Standard libraries
import time
import os
import webbrowser

# Third-party libraries
import numpy as np
import tensorflow as tf

# Own modules
import settings as st
from datasets import Dataset

base_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(base_dir, st.SUMMARY_DIR)


# checks if there is a log dir and creates one if not
# looks for the last execution subdir and creates the next one
def get_log_subdir():
    last_exec = 0

    # deleting old log folder if present, creating new folder
    try:
        os.stat(log_dir)

        # there is already a log directory
        for filename in os.listdir(log_dir):
            try:
                ex_n = int(filename[5:])
                last_exec = max(last_exec, ex_n)
            except:
                continue
    except:
        # there is already a log directory
        os.mkdir(log_dir)

    log_subdir = os.path.join(log_dir, 'exec_' + str(last_exec + 1))

    return log_subdir, last_exec + 1

# data
ds = Dataset(base_dir + '/aux_files/iris_dataset.csv', 'classification', 'iris_dataset')
ds.to_one_hot_encoding()
ds.normalise()

# splitting dataset: 60% training, 20% validation (implicit), 20% test.
# (90, 30, 30) examples. No cross validation.
ds.split(0.6, 0.2)

# graph definition
h_l = settings.HIDDEN_LAYERS

if not h_l:
    raise ValueError()

activation_f = tf.nn.sigmoid
n_classes = ds.get_n_classes()
n_features = ds.get_n_features()
layers = [n_features] + h_l + [n_classes]

# input layer
# each minibatch is feed as an array where every column is an example and the
# number of them is not fixed (thus the None in the shape)
input_ = tf.placeholder(tf.float32, shape=[n_features, None], name='l1/a')

# lists of weights and biases ops
op_weights, op_biases, op_outputs = [None], [None], [input_]

for i, n in list(enumerate(layers))[1:]:
    with tf.name_scope('l' + str(i + 1) + '/'):
        # weight initialisation: minimising total variance to avoid saturation
        op_normal = tf.random_normal([n, layers[i - 1]], stddev=np.sqrt(n), seed=st.SEED)
        op_weights.append(tf.Variable(op_normal, name='w'))

        op_bias_normal = tf.truncated_normal([n, 1], seed=st.SEED)
        op_biases.append(tf.Variable(op_bias_normal, name='b'))

        activation = tf.matmul(op_weights[-1], op_outputs[-1]) + op_biases[-1]
        op_outputs.append(activation_f(activation, name='o'))

# target
y_ = tf.placeholder(tf.float32, shape=[n_classes, None], name='y_')

# loss
output = op_outputs[-1]

with tf.name_scope('loss/'):
    loss = - tf.reduce_mean(tf.reduce_sum(y_ * tf.log(output) + (1 - y_) * tf.log(1 - output), reduction_indices=0))

    if st.L2_PARAM > 0:
        for w in op_weights[1:]:
            loss += st.L2_PARAM / 2 * tf.reduce_sum(tf.square(w))

# evaluation
with tf.name_scope('accuracy/'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 0), tf.argmax(output, 0)), tf.float32))

# training
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = tf.train.GradientDescentOptimizer(st.LEARNING_RATE).minimize(loss, global_step= global_step)

# variable initialisation op
init = tf.global_variables_initializer()

subdir, exec_number = get_log_subdir()

t_summary_writer = tf.train.SummaryWriter(log_dir + '/exec_{}/train'.format(exec_number), tf.get_default_graph())
v_summary_writer = tf.train.SummaryWriter(log_dir + '/exec_{}/validation'.format(exec_number), tf.get_default_graph())
te_summary_writer = tf.train.SummaryWriter(log_dir + '/exec_{}/test'.format(exec_number), tf.get_default_graph())

with tf.name_scope('summaries/'):
    tf.scalar_summary('loss', loss)
    tf.scalar_summary('accuracy', accuracy)
    summary_op = tf.merge_all_summaries()

# execution
sess = tf.Session()

with sess.as_default():

    inputs_v, targets_v = ds.get_validation_data(transpose=True)
    inputs_te, targets_te = ds.get_test_data(transpose=True)

    # initialising variables
    init.run()

    step = 0
    n_unimproved_steps = 0
    best_loss_v = float("inf")

    t_start = time.clock()

    while n_unimproved_steps < st.STEPS:
    # while global_step.eval() < 5000:
        inputs_t, targets_t = ds.get_minibatch(st.MINI_BATCH_SIZE, autoreset=True, transpose=True)
        t, loss_t, acc_t, summary = sess.run([train_op, loss, accuracy, summary_op],
                                             feed_dict={input_: inputs_t, y_: targets_t})
        step = global_step.eval()
        t_summary_writer.add_summary(summary, step)

        # summaries and information for tensorboard
        loss_v, acc_v, summary = sess.run([loss, accuracy, summary_op],
                                          feed_dict={input_: inputs_v, y_: targets_v})
        v_summary_writer.add_summary(summary, step)

        # updating validation rule
        if loss_v < best_loss_v:
            best_loss_v = loss_v
            n_unimproved_steps = 0
        else:
            n_unimproved_steps += 1

        print('Accuracy: {:<7.2%} (t), {:<7.2%} (v); '
              'Loss: {:<6.5} (t), {:<6.5} (v)'.format(acc_t, acc_v, loss_t, loss_v))

    t_end = time.clock()

    loss_te, acc_te, summary = sess.run([loss, accuracy, summary_op],
                                        feed_dict={input_: inputs_te, y_: targets_te})
    te_summary_writer.add_summary(summary, step)

    print('\n{} steps taken. Elapsed time: {:.3}s.'
          '\n\tAccuracy on validation set: {:.2%}'
          '\n\tAccuracy on previously unseen test set: {:.2%}'.
          format(global_step.eval(), t_end - t_start, acc_v, acc_te))

    print('\nDo you want to open the gathered summaries in your browser?')
    if input('"yes" or "y" if so, anything else otherwise: ').lower() in ['yes', 'y']:
        print('\n/!\ The local server may take a few seconds to load\n'
                '    Press Ctrl-C to quit\n')
        webbrowser.open_new('http://localhost:6006')
        os.system('tensorboard --logdir=' + log_dir + ' > /dev/null 2>&1')

sess.close()
