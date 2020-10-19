import numpy as np
import pandas as pd
import os
import argparse

import stcl


if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--contrast", type=int, default=100, help="contrast for the dataset")
    parser.add_argument("-t", "--tap", type=int, default=8, help="number of taps")
    # clustering param
    parser.add_argument("-d", "--dim", type=int, default=2, help="dimension used for clustering")

    # read arguments from the command line
    args = parser.parse_args()

    print('contrast is {}.'.format(args.contrast))
    print("number of tap is {}.".format(args.tap))
    print("dimension for clustering is is {}.".format(args.dim))

    # load experimental info
    print('loading experimental info')
    data_path = 'data/gaussian_stim_data'
    info = pd.read_csv(os.path.join(data_path, 'contrast{}_sta.csv'.format(args.contrast)))

    # load data
    print("loading data...")
    # load stim and spike data

    data = np.load(os.path.join(data_path, 'contrast{}.npz'.format(args.contrast)))

    stim = data['stim']
    if stim.shape[0] > stim.shape[1]: # 0st dim should be spatial, 1st dim should be time
        stim = stim.T

    spike_counts = data['spike_counts']

    save_folder_name = "results/gaussian_stim_contrast{}_tap{}_cluster_dim{}".format(args.contrast, args.tap, args.dim)
    if not os.path.exists(save_folder_name):
        os.makedirs(save_folder_name)

    stcl.run_stcl(stim, spike_counts, info, tap=args.tap, cluster_dim=args.dim, save_folder_name=save_folder_name)
