import argparse
import numpy as np
import logging
import sys
import os

from lib import fnameRW


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', '-dir', type=str, default='',
                        help='file to be tested')
    args = parser.parse_args()
    test_files = os.listdir(args.directory)

    result_dict = {}
    models = ['n', 't', 'd', 'c']
    kgs = ['w11', 'f13']
    for model in models:
        for kg in kgs:
            result_dict[(model, kg)] = [0, 0]

    for test_file in test_files:
        train_args_dict = fnameRW.fname_read(test_file)
        key = (train_args_dict['model'], train_args_dict['kg_choice'])
        f = open(args.directory + '/' + test_file, 'r')
        last_line = f.readlines()[-1]
        dev, test = last_line.split()
        dev = float(dev)
        test = float(test)
        if result_dict[key][0] < dev:
            result_dict[key] = [dev, test]
            print(result_dict)


if __name__ == '__main__':
    main()
