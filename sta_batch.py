import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import pysta

import os
import argparse

def do_sta(spike_triggered_stim_all_channels, spike_count_all_channels=None, info=None, folder_name=None):
    sigs = list()
    num_samples = list()

    for ch_idx in tqdm(range(len(spike_triggered_stim_all_channels))):
        channel_name = info["channel_names"][ch_idx]
        #print(channel_name)

        if spike_count_all_channels is None:
            # print("perform simple mean")
            sta = np.mean(spike_triggered_stim_all_channels[ch_idx], axis=0)    # simple average
        else:
            # print("perform weighted average")
            sta = np.average(spike_triggered_stim_all_channels[ch_idx], weights=spike_count_all_channels[ch_idx], axis=0)  # simple average
        num_samples.append(spike_triggered_stim_all_channels[ch_idx].shape[0])

        # calc mean and std of sta
        m = np.mean(sta.reshape([-1, 1]))
        sig = np.std(sta.reshape([-1, 1]))
        # print(m, sig)
        sigs.append(sig)
        # plt.hist(sta.reshape([-1,1]))
        # plt.title("mean={:.2f}, std={:.2f}".format(m,sig))

        if folder_name is not None:
            if info is not None:
                pysta.plot_stim_slices(sta, 8, 8, dt=1000/info["sampling_rate"])
            else:
                pysta.plot_stim_slices(sta, 8, 8)
            plt.savefig("{}/{}_sta.png".format(folder_name, channel_name))
            plt.close()

    if folder_name is not None:
        sigs = np.array(sigs)
        num_samples = np.array(num_samples)
        np.savetxt("{}/sta_sigs.txt".format(folder_name), sigs)
        np.savetxt("{}/num_samples.txt".format(folder_name), num_samples)


if __name__ == '__main__':

    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset name")
    parser.add_argument("-t", "--tap", type=int, default=8, help="number of taps")

    # read arguments from the command line
    args = parser.parse_args()

    # get dataset name
    if args.dataset:
        dataset = args.dataset
    else:
        print("provide dataset name!")
        exit(-1)

    print("number of tap is {}.".format(args.tap))

    # load data
    print("loading data...")
    # load stim and spike data
    stim, spike_train, info = pysta.load_data(dataset, "data")
    num_channels = spike_train.shape[0]
    # print(info["channel_names"])

    # grab spike-triggered stim
    spike_triggered_stim_all_channels, spike_count_all_channels = pysta.grab_spike_triggered_stim_all_channels(stim,
                                                                                                               spike_train,
                                                                                                               args.tap)

    # do STA
    folder_name = "{}_tap{}_sta".format(dataset, args.tap)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    do_sta(spike_triggered_stim_all_channels, spike_count_all_channels, info, folder_name) # weighted average
    #do_sta(spike_triggered_stim_all_channels)  # simple mean
