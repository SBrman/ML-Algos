#! python3

__author__ = "Simanta Barman"
__email__  = "barma017@umn.edu"

import random
import torch.nn as nn
from nlp2_preprocessor import *
from nlp2_rnns import *


class Model:
    def __init__(self, n_letters, n_hidden, n_layers, dropout,
                 all_categories=ALL_CATEGORIES, category_lines=CATEGORY_LINES, 
                 learning_rate=0.0005, Net=RNN_0):
        
        self.all_categories = all_categories
        self.category_lines = category_lines
        self.n_categories = len(all_categories)
        self.n_layers = n_layers

        self.Net = Net(n_letters, n_hidden, n_layers, self.n_categories, dropout=dropout)
        self.criterion = nn.NLLLoss()
        self.learning_rate = learning_rate
        
        self.all_losses = []
        self.confusion_matrix = torch.zeros(self.n_categories, self.n_categories)
        
    def categoryFromOutput(self, output):
        _, argmax = output.topk(1)
        category_i = argmax[0].item()
        return self.all_categories[category_i], category_i

    @staticmethod
    def categoryTensor(category):
        """One-hot vector for category"""
        li = ALL_CATEGORIES.index(category)
        tensor = torch.zeros(1, N_CATEGORIES)
        tensor[0][li] = 1
        return tensor

    @staticmethod
    def inputTensor(line):
        """One-hot matrix of first to last letters (not including EOS) for input"""
        tensor = torch.zeros(len(line), 1, N_LETTERS)
        for li in range(len(line)):
            letter = line[li]
            tensor[li][0][ALL_LETTERS.find(letter)] = 1
        return tensor

    @staticmethod
    def targetTensor(line):
        """LongTensor of second letter to end (EOS) for target"""
        letter_indexes = [ALL_LETTERS.find(line[li]) for li in range(1, len(line))]
        letter_indexes.append(N_LETTERS - 1) # EOS
        return torch.LongTensor(letter_indexes)
    
    @staticmethod
    def randomChoice(l):
        return l[random.randint(0, len(l) - 1)]

    def randomTrainingPair(self):
        category = self.randomChoice(self.all_categories)
        line = self.randomChoice(self.category_lines[category])
        return category, line
    
    def randomTrainingExample(self):
        category, line = self.randomTrainingPair()
        category_tensor = self.categoryTensor(category)
        input_line_tensor = self.inputTensor(line)
        target_line_tensor = self.targetTensor(line)
        return category_tensor, input_line_tensor, target_line_tensor 
    
    def train(self, category_tensor, inline_tensor, outline_tensor):
        
        outline_tensor.unsqueeze_(-1)
        
        x_size = inline_tensor[0].size(0)
        hidden = self.Net.initHidden(x_size)
        
        self.Net.zero_grad()
        loss = 0
        for i in range(inline_tensor.size(0)):
            output, hidden = self.Net(category_tensor, inline_tensor[i], hidden)
            loss += self.criterion(output, outline_tensor[i])
            
        loss.backward()
        
        for p in self.Net.parameters():
            p.data.add_(p.grad.data, alpha=-self.learning_rate)
            
        return output, loss.item() / inline_tensor.size(0)
    
    def trainModel(self, n_iters, print_interval, plot_interval):
        total_loss = 0
        self.all_losses = []
        
        for k in range(1, n_iters+1):
            output, loss = self.train(*self.randomTrainingExample())
            total_loss += loss
            
            if k % print_interval == 0:
                print('(%d %d%%) %.4f' % (k, k / n_iters * 100, loss))

            if k % plot_interval == 0:
                self.all_losses.append(total_loss / plot_interval)
                total_loss = 0
                
    def sample(self, category, start_letter='A', max_length=20):
        """Sample from a category and starting letter"""
        with torch.no_grad():  # no need to track history in sampling
            category_tensor = self.categoryTensor(category)
            inTensor = self.inputTensor(start_letter)

            x_size = inTensor[0].size(0)
            hidden = self.Net.initHidden(x_size)

            output_name = start_letter

            for i in range(max_length):
                output, hidden = self.Net(category_tensor, inTensor[0], hidden)
                topv, topi = output.topk(1)
                topi = topi[0][0]
                if topi == N_LETTERS - 1:
                    break
                else:
                    letter = ALL_LETTERS[topi]
                    output_name += letter
                inTensor = self.inputTensor(letter)

            return output_name

    def samples(self, category, start_letters='ABC'):
        """Get multiple samples from one category and multiple starting letters"""
        for start_letter in start_letters:
            print(self.sample(category, start_letter))
    
    

if __name__ == "__main__":
    n_hidden = 256 
    n_layers = 1
    dropout = 0.1
    models = []
    for net in [RNN_0, RNN_1, RNN_GRU]:
        model = Model(N_LETTERS, n_hidden, n_layers, Net=net, dropout=dropout)
        model.trainModel(n_iters=100000, print_interval=5000, plot_interval=1000)
        models.append(model)
    
    
