#! python3
"""https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html"""

__author__ = "Simanta Barman"
__email__  = "barma017@umn.edu"

import random
import torch.nn as nn

from matplotlib import pyplot as plt
from matplotlib import ticker as ticker

from nlp1_preprocessor import *
from nlp1_rnns import *


class Model:
    def __init__(self, n_letters, n_hidden, n_layers, 
                 all_categories=ALL_CATEGORIES, category_lines=CATEGORY_LINES, 
                 learning_rate=0.005, Net=RNN_0):
        
        self.all_categories = all_categories
        self.category_lines = category_lines
        self.n_categories = len(all_categories)
        self.n_layers = n_layers

        self.Net = Net(n_letters, n_hidden, n_layers, self.n_categories)
        self.criterion = nn.NLLLoss()
        self.learning_rate = learning_rate
        
        self.all_losses = []
        self.confusion_matrix = torch.zeros(self.n_categories, self.n_categories)
        
    def categoryFromOutput(self, output):
        _, argmax = output.topk(1)
        category_i = argmax[0].item()
        return self.all_categories[category_i], category_i
    
    def randomChoice(self, l):
        return l[random.randint(0, len(l) - 1)]

    def randomTrainingExample(self):
        category = self.randomChoice(self.all_categories)
        line = self.randomChoice(self.category_lines[category])
        category_tensor = torch.tensor([self.all_categories.index(category)], dtype=torch.long)
        line_tensor = lineToTensor(line)
        return category, line, category_tensor, line_tensor
    
    def train(self, category_tensor, line_tensor):
        x_size = line_tensor[0].size(0)
        hidden = self.Net.initHidden(x_size)
        
        self.Net.zero_grad()
        for i in range(line_tensor.size()[0]):
            output, hidden = self.Net(line_tensor[i], hidden)
            
        loss = self.criterion(output, category_tensor)
        loss.backward()
        
        for p in self.Net.parameters():
            p.data.add_(p.grad.data, alpha=-self.learning_rate)
            
        return output, loss.item()
    
    def trainModel(self, n_iters, print_interval, plot_interval):
        current_loss = 0
        self.all_losses = []
        
        for k in range(1, n_iters+1):
            category, line, category_tensor, line_tensor = self.randomTrainingExample()
            output, loss = self.train(category_tensor, line_tensor)
            current_loss += loss
            
            if k % print_interval == 0:
                guess, _ = self.categoryFromOutput(output)
                correct = '✓' if guess == category else f'✗ ({category})'
                print(f'iteration={k}\t{k*100/n_iters}\t{loss=}\t{line=}\t{guess=}\t{correct=}')
                
            if k % plot_interval == 0:
                self.all_losses.append(current_loss / plot_interval)
                current_loss = 0
                
    def evaluate(self, line_tensor):
        x_size = line_tensor[0].size(0)
        hidden = self.Net.initHidden(x_size)
        for i in range(line_tensor.size()[0]):
            output, hidden = self.Net(line_tensor[i], hidden)
        return output
    
    def save_confusion(self, name):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(self.confusion_matrix.numpy())
        fig.colorbar(cax)

        ax.set_xticklabels([''] + self.all_categories, rotation=90)
        ax.set_yticklabels([''] + self.all_categories)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.savefig(f'confusion_{name}.png', dpi=300)

    def _create_confusion_matrix(self, name, n_confusion=10000):
        for i in range(n_confusion):
            category, line, category_tensor, line_tensor = self.randomTrainingExample()
            output = self.evaluate(line_tensor)
            guess, guess_i = self.categoryFromOutput(output)
            category_i = self.all_categories.index(category)
            self.confusion_matrix[category_i][guess_i] += 1
        
        for i in range(self.n_categories):
            self.confusion_matrix[i] = self.confusion_matrix[i] / self.confusion_matrix[i].sum()
            
        self.save_confusion(name)
        
    def predict(self, input_line, n_predictions=3):
        print(f'\n> {input_line}')
        with torch.no_grad():
            output = self.evaluate(lineToTensor(input_line))

            # Get top N categories
            topv, topi = output.topk(n_predictions, 1, True)
            predictions = []

            for i in range(n_predictions):
                value = topv[0][i].item()
                category_index = topi[0][i].item()
                print('(%.2f) %s' % (value, self.all_categories[category_index]))
                predictions.append([value, self.all_categories[category_index]])
    

if __name__ == "__main__":
    n_hidden = 256 
    n_layers = 2
    models_problem1a = []
    for name, net in zip(['RNN_0', 'RNN_1', 'GRU', 'LSTM'], [RNN_0, RNN_1, RNN_GRU, RNN_LSTM]):
        model = Model(N_LETTERS, n_hidden, n_layers, Net=net)
        model.trainModel(n_iters=100000, print_interval=5000, plot_interval=1000)
        models_problem1a.append(model)
        
        print(f"{name}'s best loss = {min(model.all_losses)}")
        
        plt.figure()
        plt.plot(model.all_losses)
        plt.xlabel('Iteration (x1000)')
        plt.ylabel('Loss')
        plt.suptitle(f'Loss vs Iterations ({name})')
        plt.savefig(f'loss1a_{name}.png', dpi=300)
        
        model._create_confusion_matrix(name)
    
