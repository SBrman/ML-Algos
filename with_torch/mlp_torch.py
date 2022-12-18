#! python3

__author__ = "Simanta Barman"
__email__  = "barma017@umn.edu"


import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from os.path import join
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

np.random.seed(42)
torch.manual_seed(42)


INPUT_LAYER_NEURONS = 64
OUTPUT_LAYER_NEURONS = 10


class OptDigits(Dataset):
    """The dataset class"""

    def __init__(self, type="train"):
        self.type = type

        assert self.type == "train" or self.type == "valid" or self.type == "test"

        data = np.genfromtxt("../data/optdigits_{}.txt".format(self.type), delimiter=",")
        x = (data[:, :-1] / 16 - 0.5).astype('float32')
        y = data[:, -1].astype('long')

        # initialize data
        self.data = []
        for idx in range(x.shape[0]):
            self.data.append((x[idx], y[idx]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return {"feature": data[0], "target": data[1]}

    def collate_func(self, batch):
        feature_batch = []
        target_batch = []

        for sample in batch:
            tmp_feature, tmp_target = sample["feature"], sample["target"]
            feature_batch.append(tmp_feature)
            target_batch.append(tmp_target)

        data = dict()
        data["feature"] = torch.tensor(np.stack(feature_batch))
        data["target"] = torch.tensor(np.stack(target_batch))

        return data


class Net(nn.Module):
    def __init__(self, hidden_layer_num=2, hidden_unit_num_list=(256, 128), activation_function="Sigmoid", dropout_rate=0.5):
        super().__init__()
        
        self.hidden_layer_num = hidden_layer_num
        self.hidden_unit_num_list = hidden_unit_num_list
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate
        
        self.activations = {'relu': F.relu, 'sigmoid': torch.sigmoid, 'tanh': torch.tanh}
        self.activation_func = self.activations[self.activation_function.lower()]

        ## model architecture
        hidden_units = [INPUT_LAYER_NEURONS, *self.hidden_unit_num_list, OUTPUT_LAYER_NEURONS]

        fcs = []
        for in_shape, out_shape in zip(hidden_units, hidden_units[1:]):
            fc = nn.Linear(in_shape, out_shape)
            fcs.append(fc)

        self.fcs = nn.ModuleList(fcs)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        self._initialize_weights()

    def forward(self, x):
        for fc in self.fcs[:-1]:
            x = fc(x)
            x = self.activation_func(x)
            x = self.dropout(x)

        x = self.fcs[-1](x)
        return F.softmax(x, dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.constant_(m.bias, 0)


class Trainer(nn.Module):
    def __init__(self, batch_size=32, lr=1e-1, epoch=200, hidden_layer_num=2, hidden_unit_num_list=(128, 64),
                 activation_function="Relu", dropout_rate=0.5, check_continue_epoch=10, logging_on=False):
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.epoch = epoch
        self.hidden_layer_num = hidden_layer_num
        self.hidden_unit_num_list = hidden_unit_num_list
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate
        self.check_continue_epoch = check_continue_epoch
        self.logging_on = logging_on


        # store the accuracy of the validation, which would be used in
        self.valid_accuracy = []

        # obtain the dataset for train, valid, test
        dataset_train = OptDigits(type="train")
        dataset_valid = OptDigits(type="valid")
        dataset_test = OptDigits(type="test")

        # obtain the dataloader
        self.train_loader = DataLoader(dataset=dataset_train, batch_size=self.batch_size, shuffle=True,
            num_workers=0, collate_fn=dataset_train.collate_func, drop_last=True)

        self.valid_loader = DataLoader(dataset=dataset_valid, batch_size=self.batch_size, shuffle=False, 
            num_workers=0, collate_fn=dataset_valid.collate_func, drop_last=False)

        self.test_loader = DataLoader(dataset=dataset_test, batch_size=self.batch_size, shuffle=False,
            num_workers=0, collate_fn=dataset_test.collate_func, drop_last=False)

        assert self.hidden_layer_num == len(self.hidden_unit_num_list)

        # the instance of the network
        self.net = Net(hidden_layer_num=self.hidden_layer_num,
                       hidden_unit_num_list=self.hidden_unit_num_list,
                       activation_function=self.activation_function,
                       dropout_rate=dropout_rate)
        
        # initialize the best net for evaluating the data in test set
        self.net_best_validation = copy.deepcopy(self.net)

        # define the optimizer
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)

        # you will define your forward function
        self.criterion = nn.CrossEntropyLoss()

    def print_msg(self, message: str):
        if self.logging_on:
            print(message)

    def evaluation(self, type="valid"):
        if type == "valid":
            data_loader = self.valid_loader
        else:
            data_loader = self.test_loader

        prediction = []
        ground_truth = []

        for i, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data["feature"], data["target"]

            # do not need to calculate the gradient
            with torch.no_grad():
                if type == "valid":
                    outputs = self.net(inputs)
                else:
                    outputs = self.net_best_validation(inputs)
                prediction_label = torch.max(outputs, 1)[1]
            prediction.append(prediction_label)
            ground_truth.append(labels)

        prediction = torch.cat(prediction, dim=0)
        ground_truth = torch.cat(ground_truth, dim=0)

        accuracy = ((prediction == ground_truth).sum() / ground_truth.shape[0]).item()

        return accuracy

    def stopping_criteria(self, check_continue_epoch=10):
        if len(self.valid_accuracy) < 10:
            return False
        return self.best_validation_accuracy not in set(self.valid_accuracy[-check_continue_epoch:])
        
    def best_validation_accuracy(self):
        return max(self.valid_accuracy)

    def train(self):
        for epoch in range(self.epoch):

            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data["feature"], data["target"]
                labels = labels.long()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward pass
                outputs = self.net(inputs)

                # backward pass
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # weight updates
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 10 == 9:  # print every 10 mini-batches
                    self.print_msg(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                    running_loss = 0.0

            self.print_msg('Finished Training for Epoch {}'.format(epoch))

            # evaluate in validation set
            self.print_msg('Prepare for the evaluation on validation set')
            valid_accuracy = self.evaluation("valid")
            self.valid_accuracy.append(valid_accuracy)
            self.print_msg("The accuracy of validation set on Epoch {} is {}".format(epoch, valid_accuracy))

            # check whether it is the current best validation accuracy, if yes, save the net work
            if valid_accuracy == max(self.valid_accuracy):
                self.net_best_validation = copy.deepcopy(self.net)

            # apply stopping_criteria
            # self.stopping_criteria() returns a bool, if True, we will terminate the training procedure
            if self.stopping_criteria(self.check_continue_epoch):
                self.print_msg("Enter the stopping_criteria, the current number of epoch is {}".format(epoch))
                break


if __name__ == "__main__":
    trainer = Trainer(logging_on=True)
    trainer.train()
    test_accuracy = trainer.evaluation("test")
    print("The accuracy of test set is {}".format(test_accuracy))
