
"""

    Author: Antonio Mejias Gil (anmegi.95@gmail.com)
    Date: Early 2017
    Description:
        This file performs hyperparameter tuning on a convolutional network on the
        MNIST dataset. It sweeps a double gird with different values for lambda
        (L2 regularisation parameters) and rho (dropout keep rate, which is 
        used only in some of the architectures) and saves the results to a file.
        Execution parameters (save ID, parameter grid, architecture, etc.) are
        defined in settings.py.
        The model saved can then be loaded with runner.py in order to execute
        operations.

"""

# Standard python libraries
import time
import datetime
import pickle
import os

# Third-party libraries
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

# Own modules
import settings as st
from architectures import tfnet, lenet, vishnet


def create_model_dir(name):
    model_path = os.path.join(st.MODEL_DIR, name)
    os.mkdir(model_path)
    return model_path

print('Did you...\n'
      '\tAdjust the number of steps?\n'
      '\tAdjust the hyperparameter intervals?\n'
      '\tWrite a new model ID (or delete the previously saved ones)?\n')

archs = {
    'tfnet': tfnet,
    'lenet': lenet,
    'vishnet': vishnet,
    'mejnet': mejnet
}

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

model_id = 0
results = []

with open(st.RESULTS_DIR + '/' + st.HYPER_ID + '.txt', 'w') as f:

    # saving execution parameters to file
    f.write(2 * (80 * '*' + '\n'))
    f.write('Training {}. Sweeping hyperparameters:\n'
            '    regularisation lambda: {}\n'
            '    dropout keep rates: = {}\n'
            'Launched: {}\n\n'.format(st.NETWORK,
                                      ', '.join([str(l) for l in st.SWEEP_LAMBDAS]),
                                      ', '.join([str(k) for k in st.SWEEP_KEEP_RATES]),
                                      datetime.datetime.now().strftime('%d %B, %H:%M')))

    # double iteration for each point in the two-dimensional grid
    for l in st.SWEEP_LAMBDAS:
        for k in st.SWEEP_KEEP_RATES:

            tf.reset_default_graph()

            # Structure initialisation
            x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
            x_image = tf.reshape(x, [-1, 28, 28, 1])
            y_ = tf.placeholder(tf.int32, shape=[None, 10], name='y_')
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            output = archs[st.NETWORK](x_image, keep_prob, l)

            # Loss and training
            cross_entropy = tf.reduce_mean(slim.losses.softmax_cross_entropy(output, y_), name='cross_entropy')

            total_loss = slim.losses.get_total_loss()
            train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss)
            predicted_class = tf.argmax(output, 1, name='predicted_class')
            correct_prediction = tf.equal(predicted_class, tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

            saver = tf.train.Saver()

            print('Model: ' + str(model_id))

            sess = tf.Session()

            with sess.as_default():

                sess.run(tf.global_variables_initializer())

                t0 = time.time()

                # training for a fixed number of iterations
                for i in range(st.N_STEPS):

                    batch = mnist.train.next_batch(st.BATCH_SIZE)
                    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: k})

                    if (i + 1) % 1000 == 0:
                        print('   Step: ' + str(i + 1))

                elapsed = time.time() - t0

                # writing accuracy to file
                v_acc = accuracy.eval(feed_dict={x: mnist.validation.images,
                                                 y_: mnist.validation.labels,
                                                 keep_prob: 1.})

                f.write(80 * '*' + '\n'
                        '    Trained {} for {} iterations, l = {}, k = {}\n'
                        '    Elapsed time: {:.2} seconds. Average time per step: {:.2} seconds.\n'
                        '    Validation accuracy: {:.2%}\n'.format(st.NETWORK, st.N_STEPS, l, k,
                                                                   elapsed, elapsed / st.N_STEPS,
                                                                   v_acc))

                model_name = st.HYPER_ID + '_' + str(model_id)
                model_id += 1

                saver.save(sess, create_model_dir(model_name) + '/model')

                f.write('    Model saved successfully as {}\n\n'.format(model_name))
                print('{} run{} out of {} completed'.format(model_id, '' if model_id == 1 else 's',
                                                            len(st.SWEEP_LAMBDAS) * len(st.SWEEP_KEEP_RATES)))

                results.append(
                    {
                        'name': model_name,
                        'l': l,
                        'k': k,
                        'validation_accuracy': v_acc
                    }
                )

            sess.close()

    # saving results to file and storing TensorFlow model.
    best_model = sorted(results, key=lambda x: x['validation_accuracy'])[-1]
    res_f = 'Finished: {}\nBest model: {} (l = {}, k = {}). Validation accuracy: {:.2%}\n'
    res = res_f.format(datetime.datetime.now().strftime('%d %B, %H:%M'),
                       best_model['name'], best_model['l'], best_model['k'], best_model['validation_accuracy'])

    pickle.dump(results, open(st.RESULTS_DIR + '/' + st.HYPER_ID + '.p', 'wb'))

    f.write(res)

