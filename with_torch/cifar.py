#! python3

__author__ = "Simanta Barman"
__email__  = "barma017@umn.edu"


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm, trange


class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)
        # dropout
        self.dropout = nn.Dropout(p=.5)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flattening
        x = x.view(-1, 64 * 4 * 4)
        # fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class Net3(nn.Module):   
    def __init__(self):
        super().__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )


    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


NETS = {1: Net1, 2: Net2, 3: Net3}


class Classifier:
    def __init__(self, Net, learning_rate, batch_size, epochs):
        
        self.net = NETS[Net]()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        
        self.optimizer = optim.SGD(self.net.parameters(), lr=learning_rate)

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        
    @staticmethod
    @torch.no_grad()
    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds==labels).item() / len(preds))
    
    def process_data(self, data):
        """
        Sends the data to the GPU if available.
        """
        inputs, labels = data
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        return inputs, labels
    
    def train_step(self, data):
        inputs, labels = self.process_data(data)
        outputs = self.net(inputs)
        loss = F.cross_entropy(outputs, labels)
        return loss, self.accuracy(outputs, labels)
    
    def train(self, trainData):

        self.L, self.A = [], []

        trainloader = torch.utils.data.DataLoader(trainData, batch_size=self.batch_size, shuffle=True, num_workers=0)

        for epoch in trange(self.epochs, desc='Epochs', position=0):
##        for epoch in range(self.epochs):
            losses = []
            accuracies = []

            for i, data in tqdm(enumerate(trainloader), desc='Samples', position=1, leave=False):
##            for i, data in enumerate(trainloader):
                self.optimizer.zero_grad()
                loss, accuracy = self.train_step(data)

                losses.append(loss)
                accuracies.append(accuracy)

                loss.backward()
                self.optimizer.step()

            avg_loss = torch.stack(losses).mean()
            avg_acc = torch.stack(accuracies).mean()
            print(f'epoch = {epoch}:\t\t\t{avg_acc=}\t\t{avg_loss=}')
            
            self.L.append(avg_loss.item())
            self.A.append(avg_acc.item())

    def predict(self, testData):
        """Returns the total accuracy."""

        testloader = torch.utils.data.DataLoader(testData, batch_size=self.batch_size, shuffle=False, num_workers=0)

        accs = []
        with torch.no_grad():
            for data in testloader:
                inputs, labels = self.process_data(data)
                outputs = self.net(inputs)
                accs.append(self.accuracy(outputs, labels))

        avg_acc = torch.stack(accs).mean().item()
        print(f'Parameters={self.batch_size=}, {self.learning_rate=}, {self.epochs=}:\nAverage Accuracy = {avg_acc}')
        return avg_acc


if __name__ == "__main__":
    
    from pprint import pformat
    import torchvision
    from torchvision import transforms

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    def run(net, bs, lr, e):
        clf = Classifier(net, batch_size=bs, learning_rate=lr, epochs=e)
        clf.train(trainData=trainset)
        accuracy = clf.predict(testData=testset)
        return accuracy


    def main():
        accuracies = [] 

        with open('log.log', 'a') as f:
            for net in [1, 2, 3]:
                for batch_size in [8, 16]:
                    for learning_rate in [1e-2, 1e-3]:
                        for epochs in range(10, 21, 5):
                            accuracy = run(net, batch_size, learning_rate, epochs)
                            info = {'Net': net, 'Batch Size': batch_size, 'Learning Rate': learning_rate, 'Epochs': epochs, 'Accuracy': accuracy}
                            accuracies.append(info)
                            f.write(pformat(info, indent=4))
                            # print(accuracy)
                            if accuracy > 0.70:
                                return
    main()
