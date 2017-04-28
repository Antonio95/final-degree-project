# Final Degree Project

This repository contains the source code for my degree thesis: *Modern Tools for Neural Networks: Google's TensorFlow Library*.

If you are interested in the report, knowing how the files work or just about anything else, feel free to contact me at: anmegi.95@gmail.com


## Requirements

The project is written on python 3.5.2 and relies on the following packages:

```
numpy==1.11.3
tensorflow==0.12.0
```

## Structure

The project is organised in two blocks: **Iris** and **MNIST**

**Iris** (tensorflow_iris):

This block essentially trains a deep FFNN on the simple Iris problem, and logs information which can be later visualised with TensorBoard (such as learning curves or the TensorFlow graph). Most of the parameters (including the number of layers and units in each) are defined in a separate file, *settings.py*.

In order to execute the program, simply modify the settings file according to your needs and run:

```
python network_in_one.py
```

The program will offer to launch TensorBoard when finished, but you may do so later too by running:

```
tensorboard --logdir=<path_to_tensorboard_logs>
```

and opening a web browser to **localhost:6006**. For further information, refer to the [TensorBoard documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tensorboard/README.md)

This block also includes a module called *Datasets* which can be used independently from the rest of the project. It allows for the loading, mixing and normalisation of datasets, among other features.

**MNIST** (tensorflow_mnist):

This block trains various convolutional neural networks, performing a grid search over their hyperparameters. The file *architectures.py* defines the various networks (using [TFSlim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)) and saves the results of the search as well as the trained models. Again, the parameters and paths are defined in a *settings.py* file. Some previously calculated results and models are stored as well.

In order to perform the grid search, tune the settings file and run:

```
python tuner.py
```

The file *runner.py* loads a previously stored model and runs the desired operations in its graph. Simply execute it and follow the instructions on the console.

## Installing

This project requires no special installation

## Contributing

Although unlikely due to the nature of the project, contributions are welcome.
Just reach me by e-mail if you are interested.

## License

The code in this repository is free for anyone to use. Credit is welcome but not mandatory.
