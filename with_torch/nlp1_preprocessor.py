#! python3
from __future__ import unicode_literals, print_function, division

__author__ = "Simanta Barman"
__email__  = "barma017@umn.edu"


import os
import io
import glob
import string
import unicodedata

import torch

# Constants
ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)


def findFiles(path): 
    return glob.glob(path)


def readLines(filename):
    lines = io.open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


def unicodeToAscii(s):
    """
    Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in ALL_LETTERS)


def get_categories():
    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []

    # Read a file and split into lines
    for filename in findFiles(r'../data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)
    
    return category_lines, all_categories, n_categories


def letterToIndex(letter):
    """Find letter index from ALL_LETTERS, e.g. "a" = 0"""
    return ALL_LETTERS.find(letter)


def letterToTensor(letter):
    """Turns a letter into a <1 x N_LETTERS> Tensor"""
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


def lineToTensor(line):
    """Turn a line into a <line_length x 1 x N_LETTERS>, or an array of one-hot letter vectors"""
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


CATEGORY_LINES, ALL_CATEGORIES, N_CATEGORIES = get_categories()


if __name__ == "__main__":
    print(findFiles(r'../data/names/*.txt'))
    print(unicodeToAscii('Ślusàrski'))
    print(letterToTensor('J'))
    print(lineToTensor('Jones').size())
    
