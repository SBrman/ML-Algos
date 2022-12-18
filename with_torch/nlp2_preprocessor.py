#! python3
from __future__ import unicode_literals, print_function, division

__author__ = "Simanta Barman"
__email__  = "barma017@umn.edu"

import os
import glob
import string
import unicodedata


# Constants
ALL_LETTERS = string.ascii_letters + " .,;'-"
N_LETTERS = len(ALL_LETTERS) + 1


def findFiles(path): 
    return glob.glob(path)


def readLines(filename):
    with open(filename, encoding='utf-8') as some_file:
        return [unicodeToAscii(line.strip()) for line in some_file]


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
    
    assert n_categories != 0, 'Problem'
    
    return category_lines, all_categories, n_categories


CATEGORY_LINES, ALL_CATEGORIES, N_CATEGORIES = get_categories()


if __name__ == "__main__":
    print('# categories:', N_CATEGORIES, ALL_CATEGORIES)
    print(unicodeToAscii("O'Néàl"))
