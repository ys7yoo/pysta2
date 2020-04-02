# spike-triggered clustering
import numpy as np
# from sklearn.covariance import MinCovDet
# from math import pi
from matplotlib import pyplot as plt

# fit mixture of Gaussians
from sklearn.mixture import GaussianMixture

import os

import pysta

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


def plot_temporal_profiles(sta, group_centers, tap, dt, vmin=0, vmax=1):
    plt.figure(figsize=(15, 4))

    ax = plt.subplot(131)
    pysta.plot_temporal_profile(sta, tap, dt)
    ax.set_ylim(vmin, vmax)

    ax = plt.subplot(132)
    pysta.plot_temporal_profile(group_centers[0], tap, dt)
    ax.set_ylim(vmin, vmax)

    ax = plt.subplot(133)
    pysta.plot_temporal_profile(group_centers[1], tap, dt)
    ax.set_ylim(vmin, vmax)


def plot_centers(center, group_center, grid_T, weights=None, sta_PSNR=None, PSNRs=None, vmin=0, vmax=1):
    num_centers = len(group_center)
    plt.figure(figsize=(5*(num_centers+1),4))

    colors = ['b','r','g']

    # plot center
    ax = plt.subplot(1, num_centers+1, 1)
    plt.plot(grid_T, center.reshape([8 * 8, -1]).T, 'k', alpha=0.3)

    # PSNR = pysta.calc_PSNR(center)
    if sta_PSNR is None:
        plt.title("STA")
    else:
        plt.title("STA, PSNR={:.2f}".format(sta_PSNR))
    ax.set_ylim(vmin, vmax)


    for i in range(num_centers):
        ax=plt.subplot(1, num_centers+1, i+2)
        plt.plot(grid_T, group_center[i].reshape([8 * 8, -1]).T, colors[i], alpha=0.3)
        ax.set_ylim(vmin, vmax)

        title_string = "group {}".format(i+1)
        if weights is not None:
            title_string = title_string + ",weight={:.2f}".format(weights[i])
        if PSNRs is not None:
            title_string = title_string + ",PSNR={:.1f}".format(PSNRs[i])
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


