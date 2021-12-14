

# # Fully Connected Neural Network
Digit recognition (MNIST dataset) using a fully connected neural network.

1. [General](#General)
    - [Background](#background)
    - [Net Structure](https://github.com/tomershay100/Fully-Connected-Neural-Network/blob/main/README.md#net-structure)
    - [Running Instructions](https://github.com/tomershay100/Fully-Connected-Neural-Network/blob/main/README.md#running-instructions)
2. [Dependencies](#dependencies) 
3. [Installation](#installation)

## General

### Background
Implementation of a neural network for digit classification, on the MNIST dataset, which takes as an input a ``28*28`` grayscale image (``784`` floating point values of pixels between ``0-255``).

### Net Structure
The network has one hidden layer in size 128 (default) and it performs multiple epochs (20 by default) and trains the model by minimizing the Negative Log Likelihood (NLL).

During learning, the network verifies its accuracy on an independent set of data (about ``10%`` of the training set) on which learning is not performed. This group is called a ``validation set``. After all the epochs, the network saves its best condition, the weights that resulted the maximum accuracy on the validation set, to prevent overfitting.

Finally, the network exports a graph of the accuracy on the validation and the training sets, by the number of epochs, and verifies the accuracy on the testing set.

### Running Instructions

The program gets several arguments, and this can be seen with the ``-h`` flag when running. A total of about seven arguments can be sent:
* flag ```-train_x TRAIN_X_PATH``` for the training images file path (file that contains 784 values in each row).
* flag ```-train_y TRAIN_Y_PATH``` for the training labels file path (file that contains one value between ``0-9`` in each row and has the same rows number as the train_x file).
* flag ```-test_x TEST_X_PATH``` for the testing images file path (file that contains 784 values in each row).
* flag ```-test_y TEST_Y_PATH``` for the testing labels file path (file that contains one value between ``0-9`` in each row and has the same rows number as the train_x file).
* flag ```-lr LEARNING_RATE``` for the learning rate that will be used while training the model (``default value = 0.1``).
* flag ```-e EPOCHS``` for the number of epochs (``default value = 20``).
* flag ```-size HIDDEN_LAYER_SIZE``` for the hidden layer size (``default value = 128``).


running example:
```
	$ python3 neural_net.py -train_x train_x -train_y train_y -test_x test_x -test_y test_y -lr 0.03
```

## Dependencies
* [Python 3.6+](https://www.python.org/downloads/)
* Git
* [NumPy](https://numpy.org/install/)
* [SciPy](https://scipy.org/download/)
* [Matplotlib](https://matplotlib.org/stable/users/installing.html)
* [Argparse](https://pypi.org/project/argparse/)

## Installation

1. Open the terminal.
2. Clone the project by:
	```
	$ git clone https://github.com/tomershay100/Fully-Connected-Neural-Network.git
	```	
3. Run the ```neural_net.py``` file:
	```
	$ python3 neural_net.py -lr 0.03 -e 10 -size 100
	 ```
