# Tomer Shay
import numpy as np
import argparse
from scipy.special import softmax
from matplotlib import pyplot as plt


def sigmoid(layer):
    return 1 / (1 + np.exp(-layer))


def sigmoid_derivative(layer):
    return sigmoid(layer) * (1 - sigmoid(layer))


def one_hot(label):
    global output_layer_size
    arr = np.zeros((output_layer_size, 1))
    arr[int(label)] = 1
    return arr.copy()


def forward_prop(input_layer, all_weights, all_biases):
    hidden_layer_before_sigmoid = np.dot(all_weights[0], np.array([input_layer]).T) + all_biases[0]
    hidden_layer = sigmoid(hidden_layer_before_sigmoid)

    output_layer_before_softmax = np.dot(all_weights[1], hidden_layer) + all_biases[1]
    output_layer = softmax(output_layer_before_softmax)

    return [hidden_layer_before_sigmoid, hidden_layer, output_layer_before_softmax, output_layer]


def back_prop(input_layer, label, all_layers, all_weights, all_biases, lr):
    output_weights = all_weights[1]

    hidden_layer_before_sigmoid = all_layers[0]
    hidden_layer = all_layers[1]
    output_layer = all_layers[3]

    # loss = np.sum(-one_hot(label).T * np.log(output_layer))

    labels_arr = one_hot(label)
    output_layer_error = output_layer - labels_arr

    hidden_layer_magnitude_error = np.dot(output_weights.T, output_layer_error)
    hidden_layer_gradient_error = sigmoid_derivative(hidden_layer_before_sigmoid)
    hidden_layer_error = hidden_layer_magnitude_error * hidden_layer_gradient_error

    new_weights_1 = all_weights[0] - lr * np.dot(hidden_layer_error, np.array([input_layer]))
    new_bias_1 = all_biases[0] - lr * hidden_layer_error

    new_weights_2 = all_weights[1] - lr * np.dot(output_layer_error, hidden_layer.T)
    new_bias_2 = all_biases[1] - lr * output_layer_error

    return [new_weights_1.copy(), new_weights_2.copy()], [new_bias_1.copy(), new_bias_2.copy()]


def predict(input_layer, all_weights, all_biases):
    all_layers = forward_prop(input_layer, all_weights, all_biases)
    return np.argmax(all_layers[3])


def init_weight(neurons_number_after, neurons_number_before):
    return np.random.uniform(-0.03, 0.03, (neurons_number_after, neurons_number_before))


def make_prediction(x_values, all_weights, all_biases):
    vec = []
    for val in x_values:
        prediction = predict(val, all_weights, all_biases)
        vec.append(int(prediction))
    return vec


def check_accuracy(vec_1, vec_2):
    count = 0
    for value_1, value_2 in zip(vec_1, vec_2):
        if int(value_1) == int(value_2):
            count += 1
    return 100 * count / len(vec_1)


# get arguments
parser = argparse.ArgumentParser()

# -train_x train_x -train_y train_y -test_x test_x -test_y test_y -lr learning_rate -e epochs -size hidden_layer_size
parser.add_argument("-train_x", dest="train_x_path", default="train_x", help="train_x file path")
parser.add_argument("-train_y", dest="train_y_path", default="train_y", help="train_y file path")
parser.add_argument("-test_x", dest="test_x_path", default="test_x", help="test_x file path")
parser.add_argument("-test_y", dest="test_y_path", default="test_y", help="test_y file path")
parser.add_argument("-lr", dest="learning_rate", default="0.1", help="Learning Rate")
parser.add_argument("-e", dest="epochs", default="20", help="Epochs")
parser.add_argument("-size", dest="hidden_layer_size", help="Hidden Layer Size")

args = parser.parse_args()

# get data from the files
print("loading files..")
train_x = np.loadtxt(args.train_x_path)
train_y = np.loadtxt(args.train_y_path)
train_x /= 255  # normalize train pixels to 0 - 1
test_x = np.loadtxt(args.test_x_path)
test_y = np.loadtxt(args.test_y_path)
test_x /= 255  # normalize test pixels to 0 - 1

# shuffle train data set
print("shuffle training set..")
randomize = np.arange(len(train_x))
np.random.shuffle(randomize)
train_x = train_x[randomize]
train_y = train_y[randomize]

print("separate into validate..")
validate_percentage = 10
validate_x = train_x[:(len(train_x) * validate_percentage) // 100]
validate_y = train_y[:(len(train_y) * validate_percentage) // 100]
train_x = train_x[(len(train_x) * validate_percentage) // 100:]
train_y = train_y[(len(train_y) * validate_percentage) // 100:]

epochs = int(args.epochs)
learning_rate = float(args.learning_rate)
input_layer_size = 784
hidden_layer_size = int(args.hidden_layer_size)
output_layer_size = 10

print("init weights and bias..")
# init weights and bias between input and hidden layer
weights_1 = init_weight(hidden_layer_size, input_layer_size)
bias_1 = np.zeros((hidden_layer_size, 1))
# init weights and bias between hidden and output layer
weights_2 = init_weight(output_layer_size, hidden_layer_size)
bias_2 = np.zeros((output_layer_size, 1))

weights = [weights_1, weights_2]
biases = [bias_1, bias_2]

validate_vec = []
train_vec = []

best_acc = -1
best_weights = []
best_bias = []

print("starting epochs..")
for i in range(epochs):
    # if (i + 1) == 10:
    #     learning_rate /= 2
    # shuffle train data set
    randomize = np.arange(len(train_x))
    np.random.shuffle(randomize)
    train_x = train_x[randomize]
    train_y = train_y[randomize]
    print(f'============ EPOCH #{i + 1} ============')
    print(f'[Learning Rate]:\t\t\t{learning_rate}')
    for x, y in zip(train_x, train_y):
        layers = forward_prop(x, weights, biases)
        new_w, new_b = back_prop(x, y, layers, weights, biases, learning_rate)

        # weight update
        weights = new_w.copy()
        biases = new_b.copy()

    train_prediction = make_prediction(train_x, weights, biases)
    train_accuracy = check_accuracy(train_y, train_prediction)
    train_vec.append(train_accuracy)
    print(f'[train accuracy]:\t\t\t{"{:.2f}".format(train_accuracy)}%')

    validate_prediction = make_prediction(validate_x, weights, biases)
    validate_accuracy = check_accuracy(validate_y, validate_prediction)
    validate_vec.append(validate_accuracy)
    print(f'[validate accuracy]:\t\t{"{:.2f}".format(validate_accuracy)}%')

    if best_acc < validate_accuracy:
        best_acc = validate_accuracy
        best_weights = weights.copy()
        best_bias = biases.copy()

print("==================================")
print("learn finished. exporting plot..")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.plot(validate_vec, label="validate")
plt.plot(train_vec, label="train")
plt.legend()
plt.savefig('plot.png')
print("plot.png is ready!")

print("starting predictions on test..")
test_prediction = make_prediction(test_x, best_weights, best_bias)
test_accuracy = check_accuracy(test_y, test_prediction)
print('========= TEST ACCURACY ==========')
print(f'{"{:.2f}".format(test_accuracy)}%')
