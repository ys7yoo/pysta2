import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

import pysta
import stc

import argparse





def centering(data, weights=None):
    #center = np.mean(data, axis=0, keepdims=True)
    center = np.average(data, weights=weights, axis=0)

    data_centered = data - center

    return data_centered, center


###############################################################################
# STC
###############################################################################
def run_stc(stim, spike_train, info, spatial_smoothing_sigma=0, tap=8, cov_algorithm="classic",  save_folder_name="stc"):

    kurtosis_coef = list()
    print("Doing STC...")
    for ch_idx in tqdm(range(num_channels)):
        channel_name = info["channel_names"][ch_idx]
        # print(channel_name)

        # grab spike-triggered stim
        spike_triggered_stim, spike_count = pysta.grab_spike_triggered_stim(stim, spike_train[ch_idx, :], tap)

        data = spike_triggered_stim
        num_samples = data.shape[0]
        weights = spike_count

        if spatial_smoothing_sigma > 0:
            # spatial smoothing
            # sig = np.sqrt(0.25)
            data = pysta.smoothe_stim(data, spatial_smoothing_sigma)

        # stack data into rows
        data_row = data.reshape([num_samples, -1])

        # centering by sta
        # data_centered, center = centering(data_row, weights)

        # center on all-half vector
        dim = data_row.shape[1]
        center = 0.5*np.ones((1,dim))
        data_centered = data_row - center

        # do STC
        eig_values, eig_vectors = stc.do_stc(data_centered, weights, cov_algorithm)
        np.savetxt("{}/{}_eig_val.txt".format(save_folder_name, channel_name), eig_values)
        # np.savez_compressed("{}/{}_eig_vec.npz".format(folder_name, channel_name), eig_vectors)

        # plot STC results
        plot_stc_results(data_centered, eig_values, eig_vectors, save_folder_name, channel_name)
        # eigen_values.append(eig_values)

        # calc kurtosis of the 1st coef
        kurtosis_coef.append(calc_kurtosis(data_centered, eig_vectors))

    # save kurtosis
    np.savetxt("{}/kurtosis.txt".format(save_folder_name), np.array(kurtosis_coef))


###############################################################################
# some other helper functions
###############################################################################
from scipy.stats import kurtosis

def calc_kurtosis(data_centered, eig_vectors):
    projected = stc.project(data_centered, eig_vectors[:, 0])
    return kurtosis(projected)


def plot_stc_results(data_centered, eig_values, eig_vectors, folder_name, channel_name):

    # remove the last zero eigenvalue
    if eig_values[-1] < 1e-10:
        eig_values = eig_values[:-1]
        eig_vectors = eig_vectors[:,:-1]


    # plot eigenvalues
    np.savetxt("{}/{}_eig_values.png".format(folder_name, channel_name), eig_values)
    plt.figure(figsize=(7, 4))
    plt.plot(eig_values, 'o:')
    YLIM = plt.ylim()
    # print(XLIM)
    plt.ylim([0, YLIM[1]])
    plt.ylabel('eigenvalues')
    plt.savefig("{}/{}_eig_values.png".format(folder_name, channel_name))
    plt.close()

    # plot the 1st eigenvector of STC
    pysta.plot_stim_slices(eig_vectors[:, 0], 8, 8, -0.1, 0.1, dt=1000/info["sampling_rate"])
    plt.savefig("{}/{}_eig_vector_1st.png".format(folder_name, channel_name))
    plt.close()

    # plot the last non-zero eigenvector of STC
    pysta.plot_stim_slices(eig_vectors[:, -1], 8, 8, -0.1, 0.1, dt=1000/info["sampling_rate"])
    plt.savefig("{}/{}_eig_vector_last.png".format(folder_name, channel_name))
    plt.close()


    # project
    projected = stc.project(data_centered, eig_vectors)  # [:,0:7]

    plt.figure(figsize=(6.5, 5))
    plt.scatter(projected[:, 0], projected[:, 1], s=10, linewidths=0, color='k', alpha=0.5)
    plt.xlabel("$c_1$")
    plt.ylabel("$c_2$")
    plt.savefig("{}/{}_projected.png".format(folder_name, channel_name))
    plt.close()

    # plot histogram of projected coeff
    num_figs = 4
    plt.figure(figsize=(5*num_figs, 4))
    for i in range(num_figs):
        plt.subplot(1, num_figs, i+1)
        plt.hist(projected[:,i], 50)
        plt.xlabel("$c_{}$".format(i+1))
        plt.ylabel("count")
    plt.savefig("{}/{}_projected_hist.png".format(folder_name, channel_name))
    plt.close()


###############################################################################
# main function is here!
###############################################################################
if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset name")
    parser.add_argument("-s", "--sigma", type=float, default=0, help="sigma for spatial smoothing")
    parser.add_argument("-t", "--tap", type=int, default=8, help="number of taps")
    parser.add_argument("-c", "--cov_algorithm", default="classic", choices=["classic", "robust"], help="algorithm for calculating covariance")

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

    if args.sigma > 0:
        str_smooth = "sigma{:.3f}_".format(args.sigma)
    else:
        str_smooth = ""
    folder_name = "{}_{}tap{}_stc_{}".format(dataset, str_smooth, args.tap, args.cov_algorithm)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # run_stc(stim, spike_train, info, tap=tap, folder_name="stc_smooth")
    run_stc(stim, spike_train, info, spatial_smoothing_sigma=args.sigma, tap=args.tap, cov_algorithm=args.cov_algorithm, save_folder_name=folder_name)
