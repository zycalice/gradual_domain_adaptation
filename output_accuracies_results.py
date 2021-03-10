
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
    all_gradual_results = ['portraits', 'gaussian', 'rot_mnist_60_conv', 'dialing_rot_mnist_60_conv',
                           'portraits_noconf', 'rot_mnist_60_conv_noconf', 'gaussian_noconf',
                           'portraits_smaller_interval', 'rot_mnist_60_conv_smaller_interval', 'gaussian_smaller_interval',
                           'portraits_more_epochs', 'rot_mnist_60_conv_more_epochs', 'gaussian_more_epochs']
    output_accuracies_loss(folder, main_results, './outputs/', 'main_accuracies')
    output_accuracies_loss(folder, all_gradual_results, './outputs/', 'all_gradual_accuracies')
