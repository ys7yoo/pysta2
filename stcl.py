# spike-triggered clustering
import numpy as np
# from sklearn.covariance import MinCovDet
# from math import pi
from matplotlib import pyplot as plt

# fit mixture of Gaussians
from sklearn.mixture import GaussianMixture


def fit(feature, initial_pred=None):

    # dim = feature.shape[1]
    if initial_pred is None:
        initial_pred = feature[:,0]

    means_init = [np.mean(feature[initial_pred<=0], axis=0),np.mean(feature[initial_pred>0], axis=0)]
    gm = GaussianMixture(n_components=2, n_init=20, means_init=means_init)
    gm.fit(feature)

    print("converged=", gm.converged_)
    print("means=", gm.means_)
    print("covariances=", gm.covariances_)
    print("weights=", gm.weights_)

    return gm


def calc_centers(spike_triggered_stim_row, spike_count, pred):
    num_labels = len(set(pred))
    centers = list()

    for i in range(num_labels):
        centers.append(np.average(spike_triggered_stim_row[pred==i,:], axis=0, weights=spike_count[pred==i]))

    return centers


def plot_centers(center, group_center, grid_T, weights=None, vmin=0, vmax=1):
    num_centers = len(group_center)
    plt.figure(figsize=(6*num_centers,5))

    colors = ['b','r','g']

    # plot center
    ax = plt.subplot(1, num_centers+1, 1)
    plt.plot(grid_T, center.reshape([8 * 8, -1]).T, 'k', alpha=0.3)
    ax.set_ylim(vmin, vmax)


    for i in range(num_centers):
        ax=plt.subplot(1,num_centers+1,i+2)
        plt.plot(grid_T, group_center[i].reshape([8 * 8, -1]).T, colors[i], alpha=0.3)
        ax.set_ylim(vmin,vmax)
        if weights is None:
            plt.title("average stim group {}".format(i+1))
        else:
            plt.title("average stim group {} (weight={:.2f})".format(i + 1,weights[i]))

