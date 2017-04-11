import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import preprocessing
import sys
import scipy.io as sio
import scipy
import math

class NeuralNet:

    """
    Imput layer: x1, ... xd; xd+1 = 1
    Hidden units: h1, ... hm; hm+1 = 1
    Ouput layer: z1, ... zk

    For this homework, m=200, d=784, k=26
    """
    def __init__(self, d, m, k):
        self.d = d
        self.m = m
        self.k = k
        self.W = None
        self.V = None


    def trainNeuralNetwork(self, images, labels, params):
        """
        images: training images (X)
        labels: training labels (y)
        params: learning weight, weight decay rate lambda (l2 reg), etc

        1. Initialize V and W (be smart about this)
        2.
        while (stopping criterion)
            pick image/label pair (Xi, yi) at random from training set
            perform forward pass
            perform backward pass
            perform stochastic gradient descent update
        store V, W
        """

        """
        Glorot and Bengio 2010
        For hyperbolic tangent units (V), sample distribution (-r,r) with
        r = sqrt (6 * (fan in + fan out)^-1) )

        For sigmoid units (W), sample distribution (-r, r) with
        r = 4 * sqrt (6 * (fan in + fan out)^-1) )
        """

        # Params[0] = W learn rate
        # Params[1] = V learn rate
        # TODO: inverse scaling with iteration
        w_learn = params[0]
        v_learn = params[1]

        # Initialize V, W
        r1 = np.sqrt(6.0 / (self.d + 1 + self.m))
        r2 = 4.0*np.sqrt(6.0 / (self.m + 1 + self.k))
        
        V = np.random.uniform(-r1, r1, size = (self.m, self.d+1))
        W = np.random.uniform(-r2, r2, size = (self.k, self.m+1))
        #V = np.random.randn(self.m, self.d+1) / np.sqrt(self.d + 1)
        #W = np.random.randn(self.k, self.m+1) / np.sqrt(self.m + 1)

        # Add fictitious dimension
        images = np.concatenate((images, np.ones((images.shape[0],1))), axis=1)

        BATCH_SIZE = 1
        random_index = np.random.permutation(images.shape[0])
        epoch_size = math.floor(random_index.shape[0] * 1.0 / BATCH_SIZE)

        sys.stdout.write('Training neural net...\n')
        sys.stdout.flush()

        # Naive stopping criterion (set iterations / epochs) for now.
        # TODO: Make better.
        for i in range(500):

            index = (i % epoch_size) * BATCH_SIZE
            if index == 0:
                random_index = np.random.permutation(images.shape[0])
                w_learn = w_learn * 0.5
                v_learn = v_learn * 0.5
                print 'Epoch #', i * 1.0 / epoch_size
                sys.stdout.flush()

            x = images[random_index[index]]
            y = np.vstack(labels[random_index[index]])

            # Forward propagation

            h = np.vstack(tanh(np.matmul(V, x.T)))


            # Add fictitious dimension to hidden layer
            s1 = np.ones((self.m + 1, BATCH_SIZE))
            s1[:-1] = h

            #np.insert

            # Hot 1 encoding
            z_index = np.argmax(sigmoid(np.matmul(W, s1)))
            z_ = np.zeros((26, 1))
            z_[z_index] = 1

            # Backward propagation

            # delta W = (z - y)h^T
            # dimensions should be k x m+1 = 26 x 201
            
            delta_W = np.matmul((z_ - y), np.vstack(s1).T)

            # delta V = W^T(z - y)(1 - h^2)x
            # dimensions should be m x d+1 = 200 x 785

            # Remove fictitious dimension from W
            W_ = W[:,:-1]

            delta_V = np.matmul(W_.T, (z_ - y))
            delta_V = np.multiply(delta_V, np.vstack(1.0 - h**2))
            delta_V = np.matmul(delta_V, np.vstack(x).T)
            
            W = W - w_learn * delta_W
            V = V - v_learn * delta_V

        self.V = V
        self.W = W
        sys.stdout.write('Done\n')
        sys.stdout.flush()
        return -1

    def predictNeuralNetwork(self, images):
        """
        images: test images
        V, W: network weights (from training)

        for each test image x
            with V, W, perform forward pass
        return all predicted labels
        """
        W = self.W
        V = self.V

        # Add a fictitious dimension
        images = np.concatenate((images, np.ones((images.shape[0],1))), axis=1)

        predictions = []

        sys.stdout.write('Making predictions...\n')
        sys.stdout.flush()

        for x in images:
            h = tanh(np.matmul(V, x))

            # Add fictitious dimension to hidden layer
            s1 = np.concatenate((h, np.ones((1,))))
            z_index = np.argmax(sigmoid(np.matmul(W, s1)))
            predictions.append(z_index + 1)

        sys.stdout.write('Done\n')
        sys.stdout.flush()

        return predictions

"""
Ouput unit acitvation function (sigmoid)
"""
def sigmoid(x):
    return scipy.special.expit(x)

"""
Hidden unit activation function (tanh)
"""
def tanh(x):
    return np.tanh(x)

"""
Cross-entropy loss function
"""
def loss(z, y):
    return -(y * np.log(z) + (1 - y) * np.log(1 - z))



"""
Center and normalize data
"""
def preprocess():
    sys.stdout.write("Loading raw data...\n")
    sys.stdout.flush()
    raw = sio.loadmat("hw6_data_dist/letters_data")

    sys.stdout.write("Done\n")
    sys.stdout.write("Centering and normalizing data...\n")
    sys.stdout.flush()
    
    test_x, train_x, train_y = raw['test_x'], raw['train_x'], raw['train_y']

    test_x = preprocessing.normalize(test_x, norm='l2')
    train_x = preprocessing.normalize(train_x, norm='l2')

    test_x = preprocessing.scale(test_x)
    train_x = preprocessing.scale(train_x)

    sys.stdout.write("Done\n")
    sys.stdout.write("Shuffling data...\n")
    sys.stdout.flush()

    # Join training data and labels for shuffling
    data = np.concatenate((train_x, train_y), axis=1)

    # Shuffle
    np.random.shuffle(data)

    # Seperate to data and labels
    train_x = data[:,:-1]
    train_y = data[:,-1]

    sys.stdout.write("Done\n")
    sys.stdout.flush()
    
    return train_x, train_y, test_x

"""
Validation training n' stuff
"""
def training():
    train_x, train_y, test_x = preprocess()

    # debug
    train_x = train_x[:100]
    train_y = train_y[:100]

    train_y = hot_one(train_y)

    total_size = train_x.shape[0]
    train_size = int(total_size * 0.8)

    train_indexes = np.random.permutation(train_x.shape[0])[:train_size]
    valid_indexes = np.random.permutation(train_x.shape[0])[train_size:]

    my_brain = NeuralNet(784, 200, 26)
    my_brain.trainNeuralNetwork(train_x[train_indexes], train_y[train_indexes], [0.02, 0.02])

    train_pred = np.array(my_brain.predictNeuralNetwork(train_x[train_indexes])).astype('float')
    valid_pred = np.array(my_brain.predictNeuralNetwork(train_x[valid_indexes])).astype('float')
    test_pred = my_brain.predictNeuralNetwork(test_x)

    train_labs = train_y[train_indexes].astype('float')
    valid_labs = train_y[valid_indexes].astype('float')
    loss = 0
    for (i, pred) in enumerate(train_pred):
        if int(pred) != np.argmax(train_labs[i]) + 1:
            loss += 1
    print loss * 1.0 / train_size

    vloss = 0
    for (i, pred) in enumerate(valid_pred):
        if int(pred) != np.argmax(valid_labs[i]) + 1:
            vloss += 1
    print vloss * 1.0 / (total_size - train_size)
    """
    with open('neuralnet_pred.csv', 'w+') as f:
        f.write("Id,Category\n")
        for i, val in enumerate(test_pred):
            f.write("{},{}\n".format(i+1, int(val)))
        f.close()
    """

"""
Hot-1 encode labels
"""
def hot_one(labels):
    y = np.zeros((int(labels.shape[0]), 26))
    for (i, label) in enumerate(labels):
        y[i, int(label) - 1] = 1
    return y

training()