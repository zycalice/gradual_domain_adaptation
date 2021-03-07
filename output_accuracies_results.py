
import collections
import pickle
import numpy as np
import hashlib
import sys
from os import path

import gradual_shift_better
import models
import regularization_helps
import utils


def output_accuracies_loss(results_folder, results_types, output_folder, output_filename):
    sys.stdout = open(output_folder + output_filename + '.txt', 'w')

    for r in results_types:
        print("\n=== " + r + " ===")

        file_path = results_folder + r + '.dat'
        if path.isfile(file_path):
            gradual_shift_better.experiment_results(file_path)
        else:
            print("\n[not ready yet]")
    sys.stdout.close()


if __name__ == '__main__':
    # Main paper experiments.
    print('Outputting main paper experiments')
    folder = './saved_files/'
    main_results = ['portraits', 'gaussian', 'rot_mnist_60_conv', 'dialing_rot_mnist_60_conv']
    output_accuracies_loss(folder, main_results, './outputs/', 'main_accuracies')
