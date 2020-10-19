# spike-triggered clustering
import numpy as np
# from sklearn.covariance import MinCovDet
# from math import pi
from matplotlib import pyplot as plt

# fit mixture of Gaussians
from sklearn.mixture import GaussianMixture
from scipy.stats import kurtosis

import os
from tqdm import tqdm
import pandas as pd

import pysta
import stc


def centering(data, weights=None):
    #center = np.mean(data, axis=0, keepdims=True)
    center = np.average(data, weights=weights, axis=0)

    data_centered = data - center

    return data_centered, center


def calc_kurtosis(data_centered, eig_vectors):
    projected = stc.project(data_centered, eig_vectors[:, 0])
    return kurtosis(projected)


###############################################################################
# STC + Clustering
###############################################################################
def run(stim, spike_counts, channel_names,
        tap=8,
        cov_algorithm="classic",
        cluster_dim=2,
        results_path="clustering"):
    # stim, spike_counts, tap = args.tap, cluster_dim = args.dim, save_folder_name = save_folder_name)

    # channel_names = list()
    num_spikes = list()
    sta_p2p = list()
    sta_std = list()

    largest_eigen_values = list()
    second_largest_eigen_values = list()
    third_largest_eigen_values = list()

    converged = list()
    weight1 = list()
    weight2 = list()
    group_center_inner_product = list()
    center1_p2p = list()
    center1_std = list()
    center2_p2p = list()
    center2_std = list()

    print("Doing clustering...")
    print("Results are saved to {}".format(results_path))
    num_channels = spike_counts.shape[0]
    for ch_idx in tqdm(range(num_channels)):
        channel_name = channel_names[ch_idx]
        #channel_name = info["channel_names"][ch_idx]
        # channel_name = info["channel"][ch_idx]

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
        eig_values, eig_vectors = stc.do_stc(data_centered, weights, cov_algorithm)
        # eig_values, eig_vectors = stc.do_stc(data_centered, weights)

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
        cl = fit(projected[:, :cluster_dim])
        converged.append(cl.converged_)
        pred = cl.predict(projected[:, :cluster_dim])

        group_centers = calc_centers(data_row, spike_count, pred)

        # calc inner product of two centers
        inner_product = np.dot(group_centers[0].ravel()-center.ravel(), group_centers[1].ravel()-center.ravel())

        # calc PSNRs for the two centers
        p2p1, sig1 = pysta.calc_peak_to_peak_and_std(group_centers[0])
        p2p2, sig2 = pysta.calc_peak_to_peak_and_std(group_centers[1])

        # save clustering results to lists
        num_spikes.append(np.sum(spike_count))

        sta = np.average(data_row, weights=spike_count, axis=0)  # to compare
        p2p, sig = pysta.calc_peak_to_peak_and_std(sta)
        sta_p2p.append(p2p)
        sta_std.append(sig)

        center1_p2p.append(p2p1)
        center1_std.append(sig1)
        center2_p2p.append(p2p2)
        center2_std.append(sig2)
        weight1.append(cl.weights_[0])
        weight2.append(cl.weights_[1])
        group_center_inner_product.append(inner_product)

        # plot group_centers
        dt = 100
        grid_T = np.linspace(-tap + 1, 0, tap) * dt
        plot_centers(sta, group_centers, grid_T, cl.weights_, p2p, [p2p1, p2p2])
        #stcl.plot_centers(sta, group_centers, grid_T, cl.weights_, p2p/sig, [p2p1/sig1, p2p2/sig2])
        plt.savefig(os.path.join(results_path, "{}_centers.png".format(channel_name)))
        plt.savefig(os.path.join(results_path, "{}_centers.pdf".format(channel_name)))
        plt.close()

        pysta.plot_stim_slices(group_centers[0], dt=dt)
        plt.savefig(os.path.join(results_path, "{}_center_1.png".format(channel_name)))
        plt.close()

        pysta.plot_stim_slices(group_centers[1], dt=dt)
        plt.savefig(os.path.join(results_path, "{}_center_2.png".format(channel_name)))
        plt.close()

        # save STA and group centers
        np.savez_compressed(os.path.join(results_path, "{}.npz".format(channel_name)), sta=sta, group_centers=group_centers)

    # save channel names and weights
    pd.DataFrame({"channel_name": channel_names,
                  "num_spikes": num_spikes,
                  # "cell_type": cell_types,
                  # STA
                  "sta_p2p": sta_p2p,
                  'sta_std': sta_std,
                  # STC
                  "eig1": largest_eigen_values, "eig2": second_largest_eigen_values, "eig3": third_largest_eigen_values,
                  # clustering
                  "converged": converged,
                  "center1_p2p": center1_p2p, "center1_std": center1_std,
                  "center2_p2p": center2_p2p, "center2_std": center2_std,
                  "weight1": weight1, "weight2": weight2,
                  "inner_product": group_center_inner_product}).to_csv(os.path.join(results_path, "clusters.csv"), index=None)


def fit(feature, initial_pred=None):

    # dim = feature.shape[1]
    if initial_pred is None:
        initial_pred = feature[:,0]

    means_init = [np.mean(feature[initial_pred<=0], axis=0),np.mean(feature[initial_pred>0], axis=0)]
    gm = GaussianMixture(n_components=2, n_init=20, means_init=means_init)
    gm.fit(feature)

    # print("converged=", gm.converged_)
    # print("means=", gm.means_)
    # print("covariances=", gm.covariances_)
    # print("weights=", gm.weights_)

    return gm


def calc_centers(spike_triggered_stim_row, spike_count, pred):
    # num_labels = len(set(pred))
    centers = list()

    # if num_labels > 1:
    for i in range(2):
        idx = pred == i
        if any(idx):
            centers.append(np.average(spike_triggered_stim_row[idx, :], axis=0, weights=spike_count[idx]))
        else:
            print("nothing for group {}".format(i))
            centers.append(np.nan*np.zeros((spike_triggered_stim_row.shape[1])))
    # else: # sometimes, only one group survives
        # centers = None
        # i = pred[0]
        # centers.append(np.average(spike_triggered_stim_row[pred == i, :], axis=0, weights=spike_count[pred == i]))
        # centers.append(np.nan)

    return centers


def plot_temporal_profiles(sta, group_centers, tap, dt, vmin=0, vmax=1, titles=None):
    plt.figure(figsize=(16, 4))

    ax = plt.subplot(131)
    pysta.plot_temporal_profile(sta, tap, dt)
    ax.set_ylim(vmin, vmax)
    if titles is not None:
        plt.title(titles[0])

    ax = plt.subplot(132)
    pysta.plot_temporal_profile(group_centers[0], tap, dt)
    ax.set_ylim(vmin, vmax)
    if titles is not None:
        plt.title(titles[1])

    ax = plt.subplot(133)
    pysta.plot_temporal_profile(group_centers[1], tap, dt)
    ax.set_ylim(vmin, vmax)
    if titles is not None:
        plt.title(titles[2])


def plot_an_example(series, cluster_dim, tap=8, dt=100, temporal_profile=True, spatial_profile=False,
                    folder_name=None, file_name_prefix=""):
    if folder_name is not None:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    dataset_name = series["dataset"]
    channel_name = series["channel_name"]
    cell_type = series["cell_type"]
    inner_product = series["inner_product"]

    PSNR = series["PSNR"]
    PSNR1 = series["PSNR1"]
    PSNR2 = series["PSNR2"]

    prob0 = series["cell_type_prob0"]
    prob1 = series["cell_type_prob1"]
    prob2 = series["cell_type_prob2"]

    #     print(i, dataset_name, channel_name, cell_type, inner_product, PSNR1, PSNR2)

    centers = np.load(os.path.join("{}_tap8_cov_classic_cluster_dim{}".format(dataset_name, cluster_dim), channel_name) + ".npz")

    sta = centers['sta']
    group_centers = centers['group_centers']


    if temporal_profile:
        plt.figure(figsize=(20, 8))

        if prob1 > prob2:
            titles = ["sta (PSNR={:.1f}, prob={:.2f})".format(PSNR, prob0),
                      "1 (PSNR={:.1f}, prob={:.2f})".format(PSNR1, prob1),
                      "2 (PSNR={:.1f}, prob={:.2f})".format(PSNR2, prob2)]

            plot_temporal_profiles(sta, group_centers, tap, dt, titles=titles)
        else:
            titles = ["STA (PSNR={:.1f}, prob={:.2f})".format(PSNR, prob0),
                      "ON (PSNR={:.1f}, prob={:.2f})".format(PSNR2, prob2),
                      "OFF (PSNR={:.1f}, prob={:.2f})".format(PSNR1, prob1)]
            plot_temporal_profiles(sta, [group_centers[1], group_centers[0]], tap, dt, titles=titles)

        if folder_name is not None:
            plt.savefig(os.path.join(folder_name, file_name_prefix + "{}_{}.png".format(dataset_name, channel_name)))

    # plot spatial profile
    if spatial_profile:
        plt.figure(figsize=(20, 8))
        pysta.plot_stim_slices(sta)

        if folder_name is not None:
            plt.savefig(os.path.join(folder_name,
                                     file_name_prefix + "{}_{}_sta.png".format(dataset_name, channel_name)))

        if prob1 > prob2:
            group_idx = [0, 1]
        else:
            group_idx = [1, 0]

        for idx in group_idx:
            plt.figure(figsize=(20, 8))
            pysta.plot_stim_slices(group_centers[idx])

            if folder_name is not None:
                plt.savefig(os.path.join(folder_name,
                                         file_name_prefix + "{}_{}_center{}.png".format(dataset_name, channel_name,
                                                                                  idx)))


def plot_examples(cluster_sorted, cluster_dim, temporal_profile=True, spatial_profile=False, folder_name=None,
                  file_name_prefix=None):

    if folder_name is not None:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    for i in range(len(cluster_sorted)):
        if folder_name is not None:
            file_name_prefix = "{:02d}_".format(i)

        plot_an_example(cluster_sorted.iloc[i], cluster_dim,
                        temporal_profile=temporal_profile,
                        spatial_profile=spatial_profile,
                        folder_name=folder_name, file_name_prefix=file_name_prefix)


def plot_centers(center, group_center, grid_T, weights=None, sta_value=None, center_values=None, vmin=0, vmax=1):
    num_centers = len(group_center)
    plt.figure(figsize=(5.5*(num_centers+1),4))

    colors = ['b','r','g']

    # plot center
    ax = plt.subplot(1, num_centers+1, 1)
    plt.plot(grid_T, center.reshape([8 * 8, -1]).T, 'k', alpha=0.3)
    plt.xlabel('time to spike (ms)')
    plt.ylabel('STA')

    # remove top & right box
    # https://stackoverflow.com/a/28720127
    # ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # PSNR = pysta.calc_PSNR(center)
    if sta_value is not None:
        plt.title("{:.2f}".format(sta_value))
    ax.set_ylim(vmin, vmax)


    for i in range(num_centers):
        ax=plt.subplot(1, num_centers+1, i+2)
        plt.plot(grid_T, group_center[i].reshape([8 * 8, -1]).T, colors[i], alpha=0.3)
        plt.xlabel('time to spike (ms)')
        plt.ylabel("cluster {} center".format(i+1))

        # remove top & right box
        # https://stackoverflow.com/a/28720127
        # ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_ylim(vmin, vmax)

        # title_string = "group {}".format(i+1)
        title_string = ""
        if center_values is not None:
            title_string = title_string + "{:.1f}".format(center_values[i])
        if weights is not None:
            title_string = title_string + ", weight={:.2f}".format(weights[i])

        plt.title(title_string)


def load_centers(cluster_folder_name, channel_names, shape=[8, 8, 8]):
    sta = list()
    center0 = list()
    center1 = list()

    for i in range(len(channel_names)):
        channel_name = channel_names[i]

        # load npz file
        clusters = np.load(os.path.join(cluster_folder_name, channel_name + ".npz"))

        sta.append(clusters['sta'].reshape(shape))

        center0.append(clusters['group_centers'][0].reshape(shape))
        center1.append(clusters['group_centers'][1].reshape(shape))

    sta = np.array(sta)
    center0 = np.array(center0)
    center1 = np.array(center1)

    return sta, center0, center1


