# spike-triggered clustering
import numpy as np
# from sklearn.covariance import MinCovDet
# from math import pi
from matplotlib import pyplot as plt

# fit mixture of Gaussians
from sklearn.mixture import GaussianMixture

def fit(feature, initial_pred):

    # dim = feature.shape[1]

    means_init = [np.mean(feature[initial_pred<0], axis=0),np.mean(feature[initial_pred>0], axis=0)]
    gm = GaussianMixture(n_components=2, n_init=10, means_init=means_init)
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

def plot_centers(centers, grid_T, weights=None):
    num_centers = len(centers)
    plt.figure(figsize=(6*num_centers,5))

    colors = ['b','r','g']
    for i in range(num_centers):
        ax=plt.subplot(1,num_centers,i+1)
        plt.plot(grid_T, centers[i].reshape([8*8,-1]).T, colors[i], alpha=0.3)
        ax.set_ylim(0.2,0.8)
        if weights is None:
            plt.title("average stim group {}".format(i+1))
        else:
            plt.title("average stim group {} (weight={:.2f})".format(i + 1,weights[i]))

