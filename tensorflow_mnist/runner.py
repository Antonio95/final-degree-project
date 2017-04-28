
"""

    Author: Antonio Mejias Gil (anmegi.95@gmail.com)
    Date: Early 2017
    Description:
        This file loads a previously saved TensorFlow model and perform
        operations in the graph (such as checking the accuracy or loss). 
        It is meant to be used interactively.

"""

print('Importing TensorFlow engine...')

# Third-party libraries
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Own modules
import settings as st

model = input('[*] Enter the model you want to load: ')

print('[*] Enter the list of ops and/or tensors to run\n'
      '    Press Enter after each op or tensor name (note that tensor names must be followed by :0)\n'
      '    When you enter the last name, press Enter twice:')

# requesting nodes
nodes = []
node = 'n'
while node:
    node = input('    ')
    if node:
        nodes.append(node)

if not nodes:
    print('No nodes specified')
    exit(0)

print('    Nodes to run: ' + ', '.join(nodes))

# requesting dataset
dataset = input('[*] Enter the dataset to run the model on: t (training), v (validation) or te (test): ')

if dataset not in ['t', 'v', 'te']:
    print('Wrong dataset name')
    exit(0)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

if dataset == 't':
    data = mnist.train
elif dataset == 'v':
    data = mnist.validation
else:
    data = mnist.test

sess = tf.Session()

with sess.as_default():

    try:
        # loading the stored model
        model_path = './{}/{}'.format(st.MODEL_DIR, model)
        saver = tf.train.import_meta_graph(model_path + '/model.meta')
        saver.restore(sess, model_path + '/model')
    except:
        print('Error loading model. Are you sure the path {} is correct?'.format(model_path))
        exit(0)

    try:
        # executing the operations
        node_results = sess.run(nodes, feed_dict={'x:0': data.images,
                                                  'y_:0': data.labels,
                                                  'keep_prob:0': 1.})
    except ValueError as e:
        print(e)
        exit(0)

    print('\nResults:')
    for (n, v) in zip(nodes, node_results):
        print('{}: {}'.format(n, v))

sess.close()
