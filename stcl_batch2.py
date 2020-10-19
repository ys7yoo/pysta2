import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd

import pysta
import stc
import stcl

import argparse


def centering(data, weights=None):
    #center = np.mean(data, axis=0, keepdims=True)
    center = np.average(data, weights=weights, axis=0)

    data_centered = data - center

    return data_centered, center


###############################################################################
# STC + Clustering
###############################################################################
def run_stcl(stim, spike_counts, info, tap=8, cluster_dim=2, save_folder_name="clustering"):
    # stim, spike_counts, tap = args.tap, cluster_dim = args.dim, save_folder_name = save_folder_name)

    channel_names = list()
    num_spikes = list()
    sta_p2p = list()
    sta_std = list()

    largest_eigen_values = list()
    second_largest_eigen_values = list()
    third_largest_eigen_values = list()

    converged = list()
    weight0 = list()
    weight1 = list()
    group_center_inner_product = list()
    center0_p2p = list()
    center0_std = list()
    center1_p2p = list()
    center1_std = list()

    print("Doing clustering...")
    print("Results are saved to {}".format(save_folder_name))
    num_channels = spike_counts.shape[0]
    for ch_idx in tqdm(range(num_channels)):
        channel_name = info["channel"][ch_idx]
        # cell_type = info["cell_types"][ch_idx]
        # print(channel_name, cell_type)

        # grab spike-triggered stim
        spike_triggered_stim, spike_count = pysta.grab_spike_triggered_stim(stim, spike_counts[ch_idx, :], tap)

        data = spike_triggered_stim
        num_samples = data.shape[0]
        weights = spike_count

        # stack data into rows
        data_row = data.reshape((num_samples, -1))

        # centering by sta
        # data_centered, center = centering(data_row, weights)

        # center on all-half vector
        dim = data_row.shape[1]
        center = 0.5*np.ones((1,dim))
        data_centered = data_row - center

        # do STC
        eig_values, eig_vectors = stc.do_stc(data_centered, weights)

        largest_eigen_values.append(eig_values[0])
        second_largest_eigen_values.append(eig_values[1])
        third_largest_eigen_values.append(eig_values[2])

        # np.savetxt("{}/{}_eig_val.txt".format(save_folder_name, channel_name), eig_values)
        # np.savez_compressed("{}/{}_eig_vec.npz".format(folder_name, channel_name), eig_vectors)

        # plot STC results
        # plot_stc_results(data_centered, eig_values, eig_vectors, save_folder_name, channel_name)
        # eigen_values.append(eig_values)

        # calc kurtosis of the 1st coef
        # kurtosis_coef.append(calc_kurtosis(data_centered, eig_vectors))

        #######################################################################
        # project
        projected = stc.project(data_centered, eig_vectors)

        #######################################################################
        # now do clustering
        cl = stcl.fit(projected[:, :cluster_dim])
        converged.append(cl.converged_)
        pred = cl.predict(projected[:, :cluster_dim])

        group_centers = stcl.calc_centers(data_row, spike_count, pred)

        # calc inner product of two centers
        inner_product = np.dot(group_centers[0].ravel()-center.ravel(), group_centers[1].ravel()-center.ravel())

        # calc PSNRs for the two centers
        p2p0, sig0 = pysta.calc_peak_to_peak_and_std(group_centers[0])
        p2p1, sig1 = pysta.calc_peak_to_peak_and_std(group_centers[1])

        # save clustering results to lists
        channel_names.append(channel_name)
        num_spikes.append(np.sum(spike_count))

        sta = np.average(data_row, weights=spike_count, axis=0)  # to compare
        p2p, sig = pysta.calc_peak_to_peak_and_std(sta)
        sta_p2p.append(p2p)
        sta_std.append(sig)

        center0_p2p.append(p2p0)
        center0_std.append(sig0)
        center1_p2p.append(p2p1)
        center1_std.append(sig1)
        weight0.append(cl.weights_[0])
        weight1.append(cl.weights_[1])
        group_center_inner_product.append(inner_product)

        # plot group_centers
        dt = 100
        grid_T = np.linspace(-tap + 1, 0, tap) * dt
        stcl.plot_centers(sta, group_centers, grid_T, cl.weights_, p2p, [p2p0, p2p1])
        #stcl.plot_centers(sta, group_centers, grid_T, cl.weights_, p2p/sig, [p2p0/sig0, p2p1/sig1])
        plt.savefig(os.path.join(save_folder_name, "{}_centers.png".format(channel_name)))
        plt.savefig(os.path.join(save_folder_name, "{}_centers.pdf".format(channel_name)))
        plt.close()

        pysta.plot_stim_slices(group_centers[0], dt=dt)
        plt.savefig(os.path.join(save_folder_name, "{}_center_1.png".format(channel_name)))
        plt.close()

        pysta.plot_stim_slices(group_centers[1], dt=dt)
        plt.savefig(os.path.join(save_folder_name, "{}_center_2.png".format(channel_name)))
        plt.close()

        # save STA and group centers
        np.savez_compressed(os.path.join(save_folder_name, "{}.npz".format(channel_name)), sta=sta, group_centers=group_centers)

    # save channel names and weights
    pd.DataFrame({"channel_name": channel_names,
                  "num_spikes": num_spikes,
                  "cell_type": info["cell type"],
                  # STA
                  "sta_p2p": sta_p2p,
                  'sta_std': sta_std,
                  # STC
                  "eig1": largest_eigen_values, "eig2": second_largest_eigen_values, "eig3": third_largest_eigen_values,
                  # clustering
                  "converged": converged,
                  "center0_p2p": center0_p2p, "center0_std": center0_std,
                  "center1_p2p": center1_p2p, "center1_std": center1_std,
                  "weight0": weight0, "weight1": weight1,
                  "inner_product": group_center_inner_product}).to_csv(os.path.join(save_folder_name, "clusters.csv"), index=None)


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

    run_stcl(stim, spike_counts, info, tap=args.tap, cluster_dim=args.dim, save_folder_name=save_folder_name)
