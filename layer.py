import math
import numpy as np
from random import random, uniform
from time import time

A, B, R = 0.5, 0.6, 0.4
BATCH_SIZE = 10
EPOCH = 100
OUTPUT_BIAS = 1#0.01
LEARNING_RATE = 0.1#0.056
HIDDEN_NEURONS = [[0,0,0] for _ in range(10)]
OUTPUT_NEURON = [0 for _ in range(11)]

SIGMOID = 'Sigmoid'
RELU = 'ReLU'
ACTIVATION_FUNCTIONS = [SIGMOID, RELU]


def generate_samples(a, b, r, n):
    X, Y = [], []
    for _ in range(n):
        x1, x2 = round(random(), 4), round(random(), 4)
        y = 1 if ((x1-a)**2 + (x2-b)**2 < r**2) else 0
        X.append([1, x1, x2])
        Y.append(y)
    return (X, Y)

def generate_data(a, b, r):
    X, Y = generate_samples(a, b, r, 100)
    x_train = np.array(X)
    y_train = np.array(Y)
    
    X, Y = generate_samples(a, b, r, 100)
    x_test = np.array(X)
    y_test = np.array(Y)

    np.savez('data.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

def init_weights(hidden_neurons, output_neuron):
    try:
        weights = np.load('weights.npz')
        (hidden, output) = weights['hidden_neurons'].tolist(), weights['output_neuron'].tolist()
        for i in range(len(output_neuron)):
            output_neuron[i] = output[i]
        for i in range(len(hidden_neurons)):
            for j in range(len(hidden_neurons[0])):
                hidden_neurons[i][j] = hidden[i][j]
    except FileNotFoundError:
        for i in range(len(output_neuron)):
            output_neuron[i] = uniform(-1, 1)
        for i in range(len(hidden_neurons)):
            for j in range(len(hidden_neurons[0])):
                if i < 5:
                    hidden_neurons[i][j] = uniform(-1, 1)
                else:
                    hidden_neurons[i][j] = uniform(-1, 1)
        output = np.array(output_neuron)
        hidden = np.array(hidden_neurons)
        np.savez('weights.npz', hidden_neurons=hidden, output_neuron=output)

def describe():
    data = np.load('data.npz')
    (X_train, Y_train), (x_test, y_test) = (data['x_train'], data['y_train']), (data['x_test'], data['y_test'])
    print(X_train.shape, Y_train.shape, x_test.shape, y_test.shape)
    print(f'1 in Y_train: {np.count_nonzero(Y_train == 1)}')
    print(f'0 in Y_train: {np.count_nonzero(Y_train == 0)}')
    print(f'1 in y_test: {np.count_nonzero(y_test == 1)}')
    print(f'0 in y_test: {np.count_nonzero(y_test == 0)}')

def get_output(activation_function, hidden_neurons, output_neuron, input):
    hidden_outputs = [OUTPUT_BIAS]
    for hidden_neuron in hidden_neurons:
        net = 0
        for i in range(len(input)):
            net += hidden_neuron[i]*input[i]
        if activation_function == SIGMOID:
            output = 1/(1+math.exp(-1*net))
        if activation_function == RELU:
            output = max(net, 0)
        hidden_outputs.append(output)
    
    net = 0
    for i in range(len(hidden_outputs)):
        try:
            net += output_neuron[i]*hidden_outputs[i]
        except FloatingPointError as E:
            print(E, output_neuron[i], hidden_outputs[i])
    
    if activation_function == SIGMOID:
        output = 1/(1+math.exp(-1*net))
    if activation_function == RELU:
        output = max(net, 0)

    return (output, hidden_outputs)

def gradient_descent(activation_function, hidden_neurons, output_neuron, input, expected_output):
    output, hidden_outputs = get_output(activation_function, hidden_neurons, output_neuron, input)
    
    delta_output_weights = []
    for h in hidden_outputs:
        if activation_function == SIGMOID:
            delta_weight = LEARNING_RATE*(expected_output - output)*output*(1 - output)*h
        if activation_function == RELU:
            delta_weight = LEARNING_RATE*(expected_output - output)*h
        
        delta_output_weights.append(delta_weight)
    
    delta_hidden_weights = []
    for i in range(len(hidden_neurons)):
        output_weight = output_neuron[i]
        h = hidden_outputs[i]
        delta_weights = []
        for x in input:
            if activation_function == SIGMOID:
                delta_weight = LEARNING_RATE*(expected_output - output)*output*(1 - output)*output_weight*h*(1 - h)*x
            if activation_function == RELU:
                delta_weight = LEARNING_RATE*(expected_output - output)*output_weight*x
            delta_weights.append(delta_weight)
        delta_hidden_weights.append(delta_weights)
    return (delta_output_weights, delta_hidden_weights)

def update_weights(hidden_neurons, output_neuron, delta_hidden_weights, delta_output_weights):
    for i in range(len(output_neuron)):
        output_neuron[i] += delta_output_weights[i]
    for i in range(len(hidden_neurons)):
        for j in range(len(hidden_neurons[0])):
            hidden_neurons[i][j] += delta_hidden_weights[i][j]

def gradient_descent_over_batch(activation_function, hidden_neurons, output_neuron, batch_size, inputs, expected_outputs):
    delta_output_weights = [0 for _ in range(len(output_neuron))]
    delta_hidden_weights = [[0 for _ in neuron] for neuron in hidden_neurons]
    for e in range(batch_size):
        del_output_weights, del_hidden_weights = gradient_descent(activation_function, hidden_neurons, output_neuron, inputs[e], expected_outputs[e])
        delta_output_weights = [delta_output_weights[i] + del_output_weights[i] for i in range(len(delta_output_weights))]
        delta_hidden_weights = [[delta_hidden_weights[i][j] + del_hidden_weights[i][j] for j in range(len(delta_hidden_weights[0]))] for i in range(len(delta_hidden_weights))]
    delta_output_weights = [weight/batch_size for weight in delta_output_weights]
    delta_hidden_weights = [[weight/batch_size for weight in neuron] for neuron in delta_hidden_weights]

    update_weights(hidden_neurons, output_neuron, delta_hidden_weights, delta_output_weights)

def train(activation_function, batch_size, epoch, hidden_neurons, output_neuron):
    data = np.load('data.npz')
    (X_train, Y_train), (_, _) = (data['x_train'], data['y_train']), (data['x_test'], data['y_test'])
    steps = math.ceil(X_train.shape[0]/batch_size)

    for _ in range(epoch):
        for i in range(steps):
            gradient_descent_over_batch(activation_function, hidden_neurons, output_neuron, batch_size, X_train[i*batch_size:(i+1)*batch_size], Y_train[i*batch_size:(i+1)*batch_size])

def test(activation_function, hidden_neurons, output_neuron):
    data = np.load('data.npz')
    (_, _), (x_test, y_test) = (data['x_train'], data['y_train']), (data['x_test'], data['y_test'])

    count = 0
    for i in range(len(x_test)):
        output, ho = get_output(activation_function, hidden_neurons, output_neuron, x_test[i])
        if round(output) == y_test[i]:
            count += 1
    return count



# generate_data(A, B, R)
# describe()
count = 0
sizes = [1, 2, 4, 5, 10, 20, 25, 50, 100]
epochs = [10, 100, 1000, 2000, 10000]
lr = {}
for batch_size in sizes:
    start = time()
    BATCH_SIZE = batch_size
    for learning_rate in range(1,5001):
        if not learning_rate%1000:
            end = time()
            print(end - start, learning_rate)
        try:
            LEARNING_RATE = learning_rate/1000
            init_weights(HIDDEN_NEURONS, OUTPUT_NEURON)
            train(SIGMOID, BATCH_SIZE, EPOCH, HIDDEN_NEURONS, OUTPUT_NEURON)
            test_count = test(SIGMOID, HIDDEN_NEURONS, OUTPUT_NEURON)
            if test_count > count:
                count = test_count
                lr[f'{LEARNING_RATE}, {BATCH_SIZE}'] = count
        except Exception as E:
            print(E, LEARNING_RATE)
            break
    end = time()
    print(end - start, batch_size)
    print("MAX",lr, count)

