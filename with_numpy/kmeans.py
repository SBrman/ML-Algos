#! python3

__author__ = "Simanta Barman"
__email__  = "barma017@umn.edu"


import numpy as np


class Kmeans:
    def __init__(self, k=3):
        self.num_cluster = k
        self.center = None
        self.error_history = []

    def run_kmeans(self, X, y):
        # initialize the centers of clutsers as a set of pre-selected samples
        init_idx = np.random.choice(list(range(len(X))), 3) 
        self.center = X[init_idx]
        num_iter = 0  # number of iterations for convergence
        
        # initialize cluster assignment
        prev_cluster_assignment = np.zeros([len(X),]).astype("int")
        cluster_assignment = np.zeros([len(X),]).astype("int")
        converged = False

        # iteratively update the centers of clusters till convergence
        while not converged:

            # iterate through the samples and compute their cluster assignment (E step)
            for i in range(len(X)):
                # use euclidean distance to measure the distance between sample and cluster centers
                distances = {cluster: self.euclidian_distance(X[i], self.center[cluster]) for cluster in range(self.num_cluster)}

                # determine the cluster assignment by selecting the cluster whose center is closest to the sample
                min_distance, argmin_cluster = min((distance, cluster) for cluster, distance in distances.items())
                cluster_assignment[i] = argmin_cluster
                
            # update the centers based on cluster assignment (M step)
            for cluster in range(self.num_cluster):
                # Get the indeces of X for the given cluster
                cluster_indeces = np.where(cluster_assignment == cluster)
                # Set the center of the cluster equal to the mean of the samples assigned to that cluster
                self.center[cluster] = X[cluster_indeces].mean(axis=0)

            # compute the reconstruction error for the current iteration
            cur_error = self.compute_error(X, cluster_assignment)
            self.error_history.append(cur_error)

            # reach convergence if the assignment does not change anymore
            converged = True if (cluster_assignment==prev_cluster_assignment).sum() == len(X) else False
            prev_cluster_assignment = np.copy(cluster_assignment)
            num_iter += 1

        return num_iter, self.error_history, cluster_assignment, self.center

    def compute_error(self, X, cluster_assignment):
        # compute the reconstruction error for given cluster assignment and centers
        error = 0
        
        for n, X_n in enumerate(X):
            for k in range(self.num_cluster):
                # r_nk is defined to be 1 if nth sample's cluster assignment matches k otherwise 0
                if cluster_assignment[n] != k: 
                    continue

                # if r_nk is 1 add to error
                error += (np.linalg.norm(X_n - self.center[k]) ** 2)

        return error
    
    @staticmethod
    def euclidian_distance(a, b):
        return np.sqrt(np.sum(np.square(a - b)))

    def params(self):
        return self.center
    
    
    
if __name__ == "__main__":
    # read in data.
    data = np.genfromtxt("../data/Digits089.csv", delimiter=",")
    X = data[:, 2:]
    y = data[:, 1]

    # apply kmeans algorithms to raw data
    clf = Kmeans(k=3)
    num_iter, error_history, cluster, center = clf.run_kmeans(X, y)
    # print(num_iter, error_history, cluster, center)
