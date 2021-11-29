# This is a sample Python script.
from snorkel.labeling import labeling_function
import sys


def print_hi(name):
    print(f'Hi, {name}')
    if not sys.version_info[:2] == (3, 7):
        print("Error, I need python 3.7")
    else:
        print("All Good!")


if __name__ == '__main__':
    print_hi('AlphaBot!')
