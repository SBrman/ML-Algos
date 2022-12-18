import numpy as np


class Tree_node:
    """
    Data structure for nodes in the decision-tree
    """
    def __init__(self):
        self.feature = None
        self.class_label = None
        self.left_child = None      # No node
        self.right_child = None     # Yes node

        self.is_leaf = False  # whether or not the current node is a leaf node


class DecisionTree:
    """
    Decision tree with binary features
    """

    def __init__(self, min_entropy):
        self.min_entropy = min_entropy
        self.root = None
        
        self.classes = None
        
    def fit(self, train_x, train_y):
        # construct the decision-tree with recursion
        self.classes = np.unique(train_y)
        self.root = self.generate_tree(train_x, train_y, self.min_entropy)

    def predict(self, test_x):
        # iterate through all samples
        prediction = np.zeros([len(test_x),]).astype("int")  # placeholder

        for i in range(len(test_x)):
            # traverse the decision-tree based on the features of the current sample
            prediction[i] = self.individual_prediction(test_x[i])

        return prediction
    
    def individual_prediction(self, features):
        node = self.root
        
        while not node.is_leaf:
            decision = features[node.feature]
            node = node.left_child if decision == 0 else node.right_child

        return node.class_label

    def generate_tree(self, data, label, min_entropy, class_probs=None):
        
        # initialize the current tree node
        cur_node = Tree_node()

        # compute the node entropy
        node_entropy = self.compute_node_entropy(label)
        
        # checking other termination criteria
        label_counts = dict(zip(*np.unique(label, return_counts=True)))

        # determine if the current node is a leaf node
        if node_entropy < min_entropy or len(label_counts) <= 1:
            # determine the class label for leaf node
            if len(label_counts) == 0:
                # If no examples exists, then class is chosen randomly
                _, max_label = max((i, v) for v, i in class_probs.items())
                # max_label = np.random.choice(self.classes)
            else:
                _, max_label = max((i, v) for v, i in label_counts.items())
                
            cur_node.class_label = max_label
            cur_node.is_leaf = True
            return cur_node
        
        # select the feature that will best split the current non-leaf node
        selected_feature = self.select_feature(data, label)
        cur_node.feature = selected_feature

        # split the data based on the selected feature and start the next level of recursion
        selected_data = data[:, selected_feature]

        for label_type, child in enumerate(['left_child', 'right_child']):
            indeces = np.where(selected_data == label_type)
            filtered_data = data[indeces]
            filtered_label = label[indeces]

            node = self.generate_tree(filtered_data, filtered_label, min_entropy, class_probs=label_counts)
            setattr(cur_node, child, node)

        return cur_node

    def select_feature(self, data, label):
        # iterate through all features and compute their corresponding entropy
        best_feat = 0
        best_information_gain = 0

        for i in range(len(data[0])):
            # compute the entropy of splitting based on the selected features
            no_feature_labels = label[np.where(data[:, i] == 0)]
            yes_feature_labels = label[np.where(data[:, i] == 1)]

            cur_entropy = self.compute_split_entropy(no_feature_labels, yes_feature_labels)  
            information_gain = self.compute_node_entropy(label) + cur_entropy

            # select the feature with minimum entropy
            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_feat = i

        return best_feat

    def compute_split_entropy(self, left_y, right_y):
        # compute the entropy of a potential split, left_y and right_y are labels for the two splits
        ly, ry = len(left_y), len(right_y)
        total = ly + ry
        split_entropy = - ((ly / total) * self.compute_node_entropy(left_y) 
                        + (ry / total) * self.compute_node_entropy(right_y))
        return split_entropy

    def compute_node_entropy(self, label):
        # compute the entropy of a tree node (add 1e-15 inside the log2 when computing the entropy to prevent numerical issue)
        EPSILON = 1e-15
        _, counts = np.unique(label, return_counts=True)
        probabilities = counts / counts.sum()

        # node_entropy = 0 - sum(p * np.log2(p + EPSILON) for p in probabilities)
        node_entropy = - probabilities.dot(np.log2(probabilities + EPSILON).T)
        # node_entropy = 1 - np.sum(np.power(probabilities, 2))
        
        return node_entropy


def get_data(dataset='train'):
    dataset = np.genfromtxt(f"../data/dt_optdigits_{dataset}.txt", delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype("int")
    return X, y

if __name__ == "__main__":
    
    X, y = get_data('train')
    
    for i in [0.001, 0.01, 0.02, 0.05, 1, 2]:
        clf = DecisionTree(i)
        clf.fit(X, y)
        predictions_val = clf.predict(X)
        cur_valid_accuracy = np.count_nonzero(predictions_val.reshape(-1) == y.reshape(-1)) / len(predictions_val)
        print(cur_valid_accuracy)

