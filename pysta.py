import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd

import os

def grab_spike_triggered_stim(stim, spike, tap):
    
    num_time_bin = stim.shape[1]
    assert(stim.shape[1] == spike.shape[0])    
    
    spike_trigered_stim = list()
    spike_count = list()
    for t in range(num_time_bin):
        if t<tap:
            continue
        sp = spike[t]
        if sp > 0:
            #print(t, sp)
            
            spike_trigered_stim.append(np.array(stim[:,t-tap+1:t+1]))
            spike_count.append(sp)
            
            # spike
    
    return np.array(spike_trigered_stim), np.array(spike_count).astype(int)


def grab_spike_triggered_stim_all_channels(stim, spike_train,tap):
    spike_triggered_stim_all_ch = list()
    spike_count_all_ch = list()
    for ch_idx in range(spike_train.shape[0]):
        spike_triggered_stim, spike_count = grab_spike_triggered_stim(stim, spike_train[ch_idx,:], tap)

        spike_triggered_stim_all_ch.append(spike_triggered_stim)
        spike_count_all_ch.append(spike_count)
    
    return spike_triggered_stim_all_ch, spike_count_all_ch


## some helper functions

def convert_list_of_ascii_to_string(content):
    return ''.join(map(chr,content))

def read_mat_cell(hf, key):
    data = list()
    
    for column in hf[key]:
        row_data = []
        row_number = 0
        content = hf[column[row_number]][:]

        row_data = convert_list_of_ascii_to_string(content)
    
        data.append(row_data)

    return data


def read_mat_cell_2d(hf, key):
    data = list()
    
    for column in hf[key]:
        row_data = []
        for row_number in range(len(column)):
            content = hf[column[row_number]][:]

            row_data.append(convert_list_of_ascii_to_string(content))
    
            #row_data.append(''.join(map(unichr, f[column[row_number]][:])))   
        data.append(row_data)

    return data
    

def load_data_mat(filename):

    info = dict()
    with h5py.File(filename,'r') as hf:
        print('List of arrays in this file: \n', hf.keys())

        stim = np.array(hf.get('stim'))
        print('Shape of the array stim: ', stim.shape)

        spike_train = np.array(hf.get('spike_train'))
        print('Shape of the array spike_train: ', spike_train.shape)

        channel_names = read_mat_cell(hf, "channel_names")
        print('length of the list channel_names: ', len(channel_names))
        info["channel_names"] = channel_names

        sampling_rate = hf.get('sampling_rate')[0][0]
        print('sampling_rate: ', sampling_rate)
        info["sampling_rate"] = sampling_rate
    
    return stim, spike_train, info


def load_cell_type(dataset_name, folder_name=""):
    cell_types_df = pd.read_csv(os.path.join(folder_name, "{}_cell_type.csv".format(dataset_name)))

    return cell_types_df


def load_data(dataset_name, folder_name=""):
    with np.load(os.path.join(folder_name,dataset_name)+".npz", allow_pickle=True) as data:
        print(data.files)
        stim = data["stim"]
        spike_train = data["spike_train"]
        info = data["info"].item()
    print(stim.shape)
    print(spike_train.shape)

    # remove "ch_" from info["channel_names"]
    channel_names = [ch.replace("ch_", "") for ch in info["channel_names"]]
    info["channel_names"] = channel_names

    # load cell type and merge it to info
    channel_names_df = pd.DataFrame({"channel_name": channel_names})
    cell_types_df = load_cell_type(dataset_name, folder_name)

    merged_df = channel_names_df.merge(cell_types_df, on="channel_name", how="outer")
    merged_df.fillna('unknown', inplace=True)

    info["cell_types"] = list(merged_df["cell_type"])

    print(info)

    return stim, spike_train, info


def find_channel_index(channel_names, channel_name):
    idx_found = np.nan
    for i, name in enumerate(channel_names):
        if channel_name in name:
            idx_found = i
            break
    return idx_found


# get center of channel
def get_xy(channel_name):
    # remove reading "ch_"
    idx = channel_name.find('_')
    if idx > 0:
        channel_name = channel_name[idx + 1:]
    # print(channel_name)

    x = int(channel_name[0])
    y = int(channel_name[1])

    abcd = channel_name[2]

    return x, y

# get_xy('ch_85a')
# get_xy_channel('85a')

# input is a list
def get_XY(channel_names):
    X = list()
    Y = list()
    for channel_name in channel_names:
        x, y = get_xy(channel_name)
        X.append(x)
        Y.append(y)
    #    print()
    X = np.array(X)
    Y = np.array(Y)

    return X, Y

def plot_on_MEA(channel_names, values, perturbation_size=0.25, xticks=None, yticks=None):
    X, Y = get_XY(channel_names)

    pert = perturbation_size * np.random.randn(len(X), 2)
    plt.scatter(X + pert[:, 0], Y + pert[:, 1], c=values)

    # plt.colorbar()

    plt.axis('equal')
    plt.axis('tight')

    if xticks is None:
        plt.xticks(list(set(X)))
    else:
        plt.xticks(xticks)

    if yticks is None:
        plt.yticks(list(set(Y)))
    else:
        plt.yticks(yticks)


def plot_stim_slices(stim, width=8, height=8, vmin=0.2, vmax=0.8, dt=None):
    stim = stim.reshape([height, width, -1])

    T = stim.shape[2]

    if T > 7:
        plt.figure(figsize=(8, 4))
        for t in range(T):
            plt.subplot(2, T / 2, t + 1)
            plt.imshow(stim[:, :, t], cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
            plt.axis('off')

            if dt is not None:
                plt.title("{:.0f} ms".format(-dt*(T-t-1)))
    else:
        plt.figure(figsize=(8, 3))
        for t in range(T):
            plt.subplot(1, T, t + 1)
            plt.imshow(stim[:, :, t], cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
            plt.axis('off')

            if dt is not None:
                plt.title("{:.0f} ms".format(-dt*(T-t-1)))


def plot_histogram_by_cell_type(df, col_name, alpha=0.5):
    idx_on = df["cell_type"] == "ON"
    df.loc[idx_on, col_name].hist(alpha=alpha)

    idx_on = df["cell_type"] == "OFF"
    df.loc[idx_on, col_name].hist(alpha=alpha)

    idx_unknown = df["cell_type"] == "unknown"
    df.loc[idx_unknown, col_name].hist(alpha=alpha)

    plt.xlabel(col_name)
    plt.ylabel("count")
    plt.legend(["ON", "OFF", "unknown"])


from scipy.ndimage import gaussian_filter


def smoothe_each_slice(stim, width=8, height=8, sigma=0.25):
    original_shape = stim.shape

    if stim.ndim in [1, 2]:
        stim = stim.reshape([height, width, -1])
    else:
        assert height == stim.shape[0]
        assert width == stim.shape[1]

    smoothed = np.zeros_like(stim)

    for t in range(stim.shape[-1]):
        smoothed[:, :, t] = gaussian_filter(stim[:, :, t], sigma=sigma)

    smoothed = smoothed.reshape(original_shape)

    return smoothed


def smoothe_stim(spike_triggered_stim, sig):
    # smooth stim
    num_samples = spike_triggered_stim.shape[0]
    smoothed_spike_triggered_stim = [smoothe_each_slice(spike_triggered_stim[i, :, :], sigma=sig) for i in
                                     range(num_samples)]
    smoothed_spike_triggered_stim = np.array(smoothed_spike_triggered_stim)

    return smoothed_spike_triggered_stim


# plot ellipse using 2-dim Gaussian distribution
# from scipy document
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html

from scipy.stats import multivariate_normal


def plot_ellipse(width, height, mean, sig, color='r', ax=None):

    def pdf_standard_norm(z, dim=1):
        if dim == 1:
            return np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
        elif dim == 2:
            return np.exp(-z ** 2) / (2 * np.pi)

    xx, yy = np.mgrid[0:width:.1, 0:height:.1]
    pos = np.dstack((xx, yy))
    rv = multivariate_normal(mean, sig)

    level = [pdf_standard_norm(2.24)]  # plot 2.24-sigma (95% for 2dim)
    if ax == None:
        #         plt.contourf(xx, yy, rv.pdf(pos))
        #         plt.colorbar()
        plt.contour(xx, yy, rv.pdf(pos), level, linewidths=2, colors=color, linestyles='dashed')
    else:
        #         ax.contourf(xx, yy, rv.pdf(pos))
        # ax.colorbar()
        ax.contour(xx, yy, rv.pdf(pos), level, linewidths=2, colors=color, linestyles='dashed')


def calc_rotation_matrix(source_point, target_point=np.array((-1, 0))):
    # normalize points
    source_point = source_point[:, :2] / np.linalg.norm(source_point[:, :2])
    target_point = target_point / np.linalg.norm(target_point)

    # find cos, sin
    cos_theta = np.dot(source_point, target_point)[0]
    sin_theta = np.cross(source_point, target_point)[0]
    # print(cos_theta, sin_theta)
    # plt.plot(cos_theta, sin_theta, 'o')
    # plt.axis([0, 1, 0, 1])

    # rotation matrix
    R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    # np.dot(R, source_point.T)

    # for transposed version
    #     Rt = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
    #     np.dot(source_point,Rt)

    return R
