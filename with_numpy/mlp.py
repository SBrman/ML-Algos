#! python3
"""TODO: Fix relu"""

__author__ = "Simanta Barman"
__email__  = "barma017@umn.edu"

import numpy as np
np.random.seed(42)


def preprocess_data(data, mean=None, std=None):
    assert isinstance(data, np.ndarray), "data must be a np.ndarray"

    if mean is not None or std is not None:
        return (data - mean) / std
    else:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std[std < 1e-10] = 1                    # To avoid divide by zero error.
        normalized_data = (data - mean) / std
        return normalized_data, mean, std

def preprocess_label(label):
    # to handle the loss function computation, convert the labels into one-hot vector for training
    n = len(label)
    one_hot = np.zeros([n, 10])
    one_hot[np.arange(n), label] = 1    # sets each datapoints, label index equal to 1

    return one_hot

@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Relu(x):
    return x * (x > 0)

def drelu(x):
    return (x > 0).astype(int)

@np.vectorize
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


ACTIVATIONS = {'sigmoid': sigmoid, 'relu': Relu, 'tanh': tanh}

ACTIVATIONS_DERIVATIVES = {
    'sigmoid': lambda x: sigmoid(x) * (1 - sigmoid(x)),
    'relu': drelu,
    'tanh': lambda x: 1 - tanh(x)**2
}

def softmax(x):
    z = np.exp(x)
    return z / np.sum(z, axis=1, keepdims=True)

def softmax_derivative(x):
    return softmax(x) * (1 - softmax(x))

def cross_entropy_loss(y, r):
    return - np.sum(r * np.log(y), keepdims=True)
    
def ce_grad(y, r):
    return - r / (y + 1e-100)
    


class MLP:
    def __init__(self, num_hid, activation="Sigmoid"):
        # initialize the weights
        self.weight_1 = np.random.random([64, num_hid]) / 100
        self.bias_1 = np.random.random([1, num_hid]) / 100
        self.weight_2 = np.random.random([num_hid, 10]) / 100
        self.bias_2 = np.random.random([1, 10]) / 100

        self.activation = ACTIVATIONS[activation.lower()]
        self.activation_derivative = ACTIVATIONS_DERIVATIVES[activation.lower()]
        
    def __call__(self, x):
        return self.forward_pass(x, update=False)
        
    def forward_pass(self, x, update=True):
        """
        returns: None
        Updates the layer1, activation1, layer2, activation2 attributes.
        """
        if x.shape[0] != 64: 
            x = x.T
        
        layer1 = np.dot(x.T, self.weight_1) + self.bias_1              # (1, 64) (64, num_hid)
        activation1 = self.activation(layer1)                          # (1, num_hid)
        
        layer2 = np.dot(activation1, self.weight_2) + self.bias_2      # (1, num_hid) x (num_hid, 10)
        softmax_out = softmax(layer2)                                  # (1, 10)
        
        if update:
            self.layer1 = layer1
            self.activation1 = activation1
            self.layer2 = layer2
            self.softmax_out = softmax_out
        
        return softmax_out
        
    def backward_pass(self, x, y):
        """
        Returns the gradients calculated from all the samples.
        """
        I = np.ones(shape=(1, len(y)))

        # dE/dB_2 = (y - r).I
        y_minus_r = self.softmax_out - y
        dE_dB2 = I @ y_minus_r

        # dE/dW_2 = (y - r).h
        dE_dW2 = self.activation1.T @ y_minus_r
        
        # dE/dB_1 = (((y - r).h).w2).(1-h)
        dw1p = (self.weight_2 @ y_minus_r.T).T \
               * self.activation_derivative(self.activation1) \
               * (1 - self.activation_derivative(self.activation1))
        dE_dB1 = I @ dw1p

        # dE/dW_2 = ((((y - r).h).w2).(1-h)).x^T
        dE_dW1 = x.T @ dw1p
        
        for x, dx in zip([self.weight_1, self.weight_2, self.bias_1, self.bias_2], [dE_dW1, dE_dW2, dE_dB1, dE_dB2]):
            assert x.shape == dx.shape, f"Incompatible dimensions. {x.shape} != {dx.shape}"
        
        return dE_dW1, dE_dB1, dE_dW2, dE_dB2
    
    def fit(self, train_x, train_y, valid_x, valid_y, lr=5e-4):
        
        count = 0
        best_valid_acc = 0

        i = 0
        while count <= 100:
            i += 1
            
            # forward pass
            self.forward_pass(train_x)

            # backward pass
            dw1, db1, dw2, db2 = self.backward_pass(train_x, train_y)
            # update the corresponding parameters based on sum of gradients for above the training samples
            step_size = lr * (100/(i))                                  # Using diminishing step size rule
            self.weight_1 = self.weight_1 - step_size * dw1
            self.bias_1 = self.bias_1 - step_size * db1
            self.weight_2 =  self.weight_2 - step_size * dw2
            self.bias_2 = self.bias_2 - step_size * db2
            
            # evaluate the accuracy on the validation data
            predictions = self.predict(valid_x)
            cur_valid_acc = (predictions.reshape(-1) == valid_y.reshape(-1)).sum() / len(valid_x)
            
            # compare the current validation accuracy
            if cur_valid_acc > best_valid_acc:
                best_valid_acc = cur_valid_acc
                count = 0
            else:
                count += 1

        return best_valid_acc

    def predict(self, x):
        predictions = self(x)
        y = np.argmax(predictions, axis=1, keepdims=True)
        return y


if __name__ == "__main__":

    def load_data(split):
        data = np.genfromtxt("../data/optdigits_{}.txt".format(split), delimiter=",")
        x = data[:, :-1]
        y = data[:, -1].astype('int')
        return x, y
    
    # training data
    train_x, train_y = load_data("train")

    # validation data
    valid_x, valid_y = load_data("valid")

    # test data
    test_x, test_y = load_data("test")

    # process the data, normalize the data on the validation and test set
    train_x, mean, std = preprocess_data(train_x)
    valid_x = preprocess_data(valid_x, mean, std)
    test_x = preprocess_data(test_x, mean, std)

    # process training labels into one-hot vectors
    train_y = preprocess_label(train_y)
    
    clf = MLP(10, 'Sigmoid')
    a = clf.fit(train_x, train_y, valid_x, valid_y)
    print(a)
