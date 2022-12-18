#! python3

__author__ = "Simanta Barman"
__email__  = "barma017@umn.edu"


import numpy as np

class NaiveBayes:
    def __init__(self, X_train, y_train):
        # size of the dataset
        self.n = X_train.shape[0]
        # size of the feature vector
        self.d = X_train.shape[1]
        # size of the class set
        self.K = len(set(y_train)) 

        # shapes of the parameters
        self.psis = np.zeros([self.K, self.d]) 
        self.phis = np.zeros([self.K])
        
    def fit(self, X_train, y_train):

        # compute the parameters
        for k in range(self.K):
            X_k = X_train[y_train == k]
            self.phis[k] = self.get_prior_prob(X_k)         # prior
            self.psis[k] = self.get_class_likelihood(X_k)   # likelihood

        # clip probabilities to avoid log(0)
        self.psis = self.psis.clip(1e-14, 1-1e-14)

    def predict(self, X_test):
        # compute log-probabilities
        predictions = []
        for x in X_test:
            probabilities = {} 
            for k in range(self.K):
                psis = self.psis[k][np.where(x == 1)]
                one_minus_psis = 1 - self.psis[k][np.where(x == 0)]
                probabilities[k] = np.exp(np.sum(np.log(psis)) + np.sum(np.log(one_minus_psis))) * self.phis[k]

            _, prediction = max((prob, k) for k, prob in probabilities.items())
            
            predictions.append(prediction)
            
        return np.array(predictions)

    def get_prior_prob(self, X):
        # compute the prior probability of class k 
        return X.shape[0] / self.n

    def get_class_likelihood(self, X):
        # estimate Bernoulli parameter theta for each feature for each class
        return np.sum(X, axis=0) / X.shape[0]

    
if __name__ == "__main__":
    
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer

    data = np.load('../data/spamNB.npy', allow_pickle=True)

    y = data[:, 0]
    X = data[:, 1]

    # split the data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    def vectorize(X_train, X_test):
        # vectorize the training set
        count_vect = CountVectorizer(binary=True, max_features=1000)
        X_train = count_vect.fit_transform(X_train).toarray()
        X_test = count_vect.transform(X_test).toarray()
        return X_train, X_test

    X_train, X_test = vectorize(X_train, X_test)

    NB = NaiveBayes(X_train, y_train)
    NB.fit(X_train, y_train)
    y_pred = NB.predict(X_test)

    print(y_pred)
