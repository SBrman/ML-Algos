#! python3

__author__ = "Simanta Barman"
__email__  = "barma017@umn.edu"

import numpy as np


class GaussianDiscriminant:
    def __init__(self, k=2, d=8, priors=None, shared_cov=False):
        # mean
        self.mean = np.zeros((k, d))
        # using class-independent covariance or not
        self.shared_cov = shared_cov 

        if self.shared_cov:
            # class-independent covariance
            self.S = np.zeros((d, d))
        else:
            # class-dependent covariance
            self.S = np.zeros((k, d, d))

        if priors is not None:
            self.p = priors
        else:
            # assume equal priors if not given
            self.p = [1.0 / k for i in range(k)]  

        self.k = k
        self.d = d
        
        self.label_indeces = {}
        self.index_labels = {}

    def get_within_class_scatter(self, X, y):
        """
        Returns the within class scatter.
        returns an np.ndarray with shape = (k, d, d)
        """
        S = np.zeros((len(self.label_indeces), X.shape[1], X.shape[1]))
        
        for i, X_i in enumerate(X):
            # Getting the index of the label of ith sample
            label_index = self.label_indeces[y[i]]

            # Getting the correct mean for the ith sample based on label
            mean_i = self.mean[label_index]

            # Compute (X - mu)(X - mu)^T and add to the covariance matrix
            x_minus_mu = (X_i - mean_i).reshape(-1, 1)
            S[label_index] += x_minus_mu.dot(x_minus_mu.T)
        
        for label, index in self.label_indeces.items():
            S[index] /= y[y == label].size

        return S
    
    def conditional_probability(self, x, given_class):
        covariance_matrix = self.S if self.shared_cov else self.S[given_class]

        denominator = ((2 * np.pi) ** (self.k / 2)) * np.sqrt(np.linalg.det(covariance_matrix))

        x_minus_mu = x - self.mean[given_class]
        inv_cov_mat = np.linalg.inv(covariance_matrix)
        numerator = np.exp(-0.5 * x_minus_mu.T.dot(inv_cov_mat.dot(x_minus_mu)))
        
        return numerator / denominator

    def fit(self, Xtrain, ytrain):
        # Setting a label index dictionary to correctly retrieve the means for each class
        self.index_labels = dict(enumerate(np.unique(ytrain)))
        self.label_indeces = {label: index for index, label in self.index_labels.items()}
        
        # Get the indeces corresponding to different labels
        indeces = {label: np.where(ytrain == label) for label in self.label_indeces}
        
        # compute the mean for each class
        for i, label in enumerate(self.label_indeces):
            self.mean[i] = Xtrain[indeces[label]].mean(axis=0)

        if self.shared_cov:
            # compute the class-independent covariance
            s = self.get_within_class_scatter(Xtrain, ytrain)
            # Summing the between class scatter to get within class scattar
            self.S = np.sum(s, axis=0)  
        else:
            # compute the class-dependent covariance
            self.S = self.get_within_class_scatter(Xtrain, ytrain)
    
    def predict(self, Xtest):
        predicted_class = np.ones(Xtest.shape[0])
        
        for i in np.arange(Xtest.shape[0]):
            class_probabilities = {}
            
            for c in np.arange(self.k):
                class_probabilities[c] = self.conditional_probability(x=Xtest[i], given_class=c) * self.p[c]

            # determine the predicted class based on the values of discriminant function
            # remember to return 1 or 2 for the predicted class
            max_probability, arg_max = max((prob, label) for label, prob in class_probabilities.items())
            predicted_class[i] = self.index_labels[arg_max]

        return predicted_class


if __name__ == "__main__":
    df = np.genfromtxt("../data/training_data_discriminant.txt", delimiter=",")
    dftest = np.genfromtxt("../data/test_data_discriminant.txt", delimiter=",")
    Xtrain = df[:, 0:8]
    ytrain = df[:, 8]
    Xtest = dftest[:, 0:8]
    ytest = dftest[:, 8]

    clf = GaussianDiscriminant(2, 8, [0.3, 0.7])
    clf.fit(Xtrain, ytrain)

    predictions = clf.predict(Xtest)
    errors = sum(abs(predictions - ytest))
    print(predictions, errors)
