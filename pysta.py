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
        if t < tap:
            continue
        # if t >= stim.shape[1]:
        #     continue
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


def clip_spike_times(spike_times, ts):
    idx = np.logical_and(spike_times > ts[0], spike_times < ts[-1])
    return spike_times[idx]


def count_spikes(spike_times, bins, timestamp_start=0):
    num_bins = bins.shape[0]
    spike_count = np.zeros_like(bins)
    for i in range(num_bins):
        if i == 0:
            continue

        spike_count[i] = np.sum(np.logical_and(spike_times >= timestamp_start + bins[i-1], spike_times < timestamp_start + bins[i]))
    return spike_count

from tqdm import tqdm
def load_gaussian_stim_data_mat(data_path, stim_type, contrast):
    all_bins = list()
    all_cell_types = list()
    all_cell_types_by_sta = list()
    all_set_numbers = list()
    all_channel_names = list()
    all_spike_counts = list()

    # read cell types
    cell_types = pd.read_csv(os.path.join(data_path, 'cell_type.csv'))
    # cell_types['cell type'].value_counts()
    # cell_types

    # read stimulus
    import scipy.io as sio
    filename = os.path.join(data_path, 'Stimulus/StimInfo_8pix_200um_{}_10Hz_15min_contrast{}'.format(stim_type, contrast))
    print(filename)
    stim = list()
    for stim_info in sio.loadmat(filename)['StimInfo']:
        stim.append(stim_info[0][0])
        # print(stim_info[0][0].shape)
    stim = np.array(stim)

    # read spike counts
    path = os.path.join(data_path, 'Spike_Timestamp/Contrast_{}'.format(contrast))

    # load time stamps
    filename = os.path.join(path, 'A1a.mat')
    print(filename)
    ts = sio.loadmat(filename)['A1a'].ravel()
    # print(ts[0], ts[-1], ts.shape)

    # load spike times
    # channel_names = list()
    spike_counts = list()
    for i, channel_name in enumerate(tqdm(cell_types['channel'])):
        filename = os.path.join(path, channel_name)
        # print(filename)
        spike_time = sio.loadmat(filename)[channel_name].ravel()

        # count spikes
        spike_counts.append(count_spikes(spike_time, ts))

    spike_counts = np.array(spike_counts)
    # bins, channel_names, spike_counts = load_spike_counts_from_folder(path, interval_in_sec, dt_in_sec)

    # for i, ch in enumerate(channel_names):
    #     all_set_numbers.append(set_no)
    #     all_channel_names.append(ch)
    #
    #     all_cell_types.append(cell_type)
    #     all_cell_types_by_sta.append(cell_type)
    #
    #     all_bins.append(bins)
    #     all_spike_counts.append(spike_counts[i])

    # # read experiment info
    # info = pd.DataFrame({  # 'contrast': contrast,
    #     'set': all_set_numbers,
    #     'channel_name': all_channel_names,
    #     'cell_type': all_cell_types,
    #     'cell_type_by_sta': all_cell_types_by_sta})

    # bins = bins.reshape((1,-1))

    return stim, spike_counts, cell_types


def load_spike_counts_from_folder(path_name):
    import os
    from glob import glob
    import scipy.io as sio
    # load time stamps
    filename = os.path.join(path_name, 'A1a.mat')
    ts = sio.loadmat(filename)['A1a'].ravel()

    # load spike times
    channel_names = list()
    spike_counts = list()
    for filename in glob(os.path.join(path_name, 'ch_*')):
        # print(filename)
        basename = os.path.basename(filename).split('.')[0]

        channel_name = basename[3:]
        channel_names.append(channel_name)
        #     print(channel_name)

        spike_time = sio.loadmat(filename)[basename].ravel()
        spike_counts.append(count_spikes(spike_time, ts))
        # spike_time_clipped = clip_spike_times(spike_time, ts)

        # count spikes
        # spike_counts.append(count_spikes(spike_time_clipped, bins))

    return channel_names, spike_counts


def load_fullfield_data_mat(data_path, contrast):
    all_cell_types = list()
    all_cell_types_by_sta = list()
    all_set_numbers = list()
    all_channel_names = list()
    all_spike_counts = list()

    # read stimulus
    import scipy.io as sio
    filename = os.path.join(data_path,'Stimulus/StimInfo_Gaussian_Uniform_{}%_5Hz_10min.mat'.format(contrast))
    stim = sio.loadmat(filename)['StimInfo'].ravel()


    # read ON and OFF cells
    for cell_type in ['ON', 'OFF']:
        for set_no in [1, 2, 3]:
            path = os.path.join(data_path, 'Spike_Timestamp/{}/{}%/set{}'.format(cell_type, contrast, set_no))
            if not os.path.isdir(path):
                continue
            if not os.path.isfile(os.path.join(path, 'A1a.mat')):
                continue

            print(path)
            channel_names, spike_counts = load_spike_counts_from_folder(path)

            for i, ch in enumerate(channel_names):
                all_set_numbers.append(set_no)
                all_channel_names.append(ch)

                all_cell_types.append(cell_type)
                all_cell_types_by_sta.append(cell_type)

                all_spike_counts.append(spike_counts[i])

    # read ON-OFF cells
    cell_type = 'ON-OFF'
    for exp_set in [1, 2, 3]:
        for cell_type_by_sta in ['ON', 'OFF', 'Unknown']:
            path = os.path.join(data_path,
                                'Spike_Timestamp/{}/STA_{}/{}%/set{}'.format(cell_type, cell_type_by_sta, contrast,
                                                                             exp_set))
            if not os.path.isdir(path):
                continue
            if not os.path.isfile(os.path.join(path, 'A1a.mat')):
                continue

            print(path)
            channel_names, spike_counts = load_spike_counts_from_folder(path)

            for i, ch in enumerate(channel_names):
                all_set_numbers.append(exp_set)
                all_channel_names.append(ch)

                all_cell_types.append(cell_type)
                all_cell_types_by_sta.append(cell_type_by_sta)

                all_spike_counts.append(spike_counts[i])

    info = pd.DataFrame({  # 'contrast': contrast,
        'set': all_set_numbers,
        'channel_name': all_channel_names,
        'cell_type': all_cell_types,
        'cell_type_by_sta': all_cell_types_by_sta})

    return stim, all_spike_counts, info


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

        # remove "ch_" from info["channel_names"]
        channel_names = [ch.replace("ch_", "") for ch in info["channel_names"]]
        info["channel_names"] = channel_names

        sampling_rate = hf.get('sampling_rate')[0][0]
        print('sampling_rate: ', sampling_rate)
        info["sampling_rate"] = sampling_rate



    return stim, spike_train, info


def load_cell_type(dataset_name, folder_name=""):
    cell_types_df = pd.read_csv(os.path.join(folder_name, "{}_cell_type.csv".format(dataset_name)))

    return cell_types_df


def load_data(dataset_name, folder_name=""):
    with np.load(os.path.join(folder_name, dataset_name)+".npz", allow_pickle=True) as data:
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

    merged_df = channel_names_df.merge(cell_types_df, on="channel_name")    # must match!
    #merged_df = channel_names_df.merge(cell_types_df, on="channel_name", how="outer")
    # merged_df.fillna('unknown', inplace=True)

    info["cell_types"] = list(merged_df["cell_type"])
    assert len(info["cell_types"]) == spike_train.shape[0]

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


def calc_grid_T(tap, dt):
    grid_T = np.linspace(-tap + 1, 0, tap) * dt

    return grid_T


def plot_temporal_profile(sta, tap, dt, ylim=None):
    # calc time to spike
    grid_T = calc_grid_T(tap, dt)

    # not plot
    plt.plot(grid_T, sta.reshape([-1, tap]).T)
    plt.xlabel('time to spike (ms)')

    if ylim is not None:
        plt.ylim(ylim)

    # remove top & right box
    # https://stackoverflow.com/a/28720127
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_stim_slices(stim, width=8, height=8, vmin=0, vmax=1, dt=None):
    stim = stim.reshape([height, width, -1])

    T = stim.shape[2]

    axes = list()

    if T > 7:
        plt.figure(figsize=(8, 4))
        for t in range(T):
            axes.append(plt.subplot(2, T / 2, t + 1))
            plt.imshow(stim[:, :, t], cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
            plt.axis('off')

            if dt is not None:
                plt.title("{:.0f} ms".format(-dt*(T-t-1)))
    else:
        plt.figure(figsize=(8, 3))
        for t in range(T):
            axes.append(plt.subplot(1, T, t + 1))
            plt.imshow(stim[:, :, t], cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
            plt.axis('off')

            if dt is not None:
                plt.title("{:.0f} ms".format(-dt*(T-t-1)))

    return axes

def plot_ellipse(avg, covariance, LINE_TYPE='r--'):

    if np.max(covariance.ravel())==0 and np.min(covariance.ravel())==0: # single pixel
        plt.plot(avg[0], avg[1], 'o'+LINE_TYPE)
        return

    theta = np.linspace(0, 2*np.pi, 100).ravel()

    circ = np.column_stack([np.cos(theta), np.sin(theta)])

    eps = 1e-9
    L = np.linalg.cholesky(covariance+np.diag([eps, eps]))
    ellipse = avg + circ @ L.T @ np.diag([2.4477, 2.4477])

    plt.plot(ellipse[:,0], ellipse[:,1], LINE_TYPE)
    return


def plot_hist_by_group(df, column, group, group_order=None, group_color=None, bins=None, density=False):
    # based on
    # https://stackoverflow.com/a/19589675
    # https://stackoverflow.com/a/39481709

    # read each group and store it to a dict
    groupby = dict()
    for group in df.groupby(group):
        groupby[group[0]] = group[1][column]

    if group_order is None:
        group_order = groupby.keys()

    values = []
    for g in group_order:
        values.append(groupby[g])

    plt.hist(values, label=group_order, color=group_color, bins=bins, density=density)
    plt.legend(loc='upper right')
    #     plt.legend(legend)
    plt.xlabel(column)

    if density:
        plt.ylabel("frequency")
    else:
        plt.ylabel("count")


def plot_hist_by_cell_type(df, col_name, bins=None, density=None):
    if len(df["cell_type"].value_counts()) == 3:
        plot_hist_by_group(df, col_name, "cell_type", ["ON", "OFF", "unknown"], group_color=["r", "b", "k"], bins=bins, density=density)
    else:
        plot_hist_by_group(df, col_name, "cell_type", ["ON", "OFF", "ON/OFF", "unknown"], group_color=["r", "b", "g", "k"], bins=bins, density=density)


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


def plot_scatter_by_group(df, col_names,
                          group_key="cell_type",
                          group_values=["ON", "OFF", "unknown"],
                          group_colors=["r","b","k"],
                          alpha=0.5,
                          loc=None):
    title = ""
    for i, type in enumerate(group_values):
        idx = df[group_key] == type
        plt.plot(df.loc[idx, col_names[0]], df.loc[idx, col_names[1]],
                    marker='o', color=group_colors[i], markeredgecolor='None',
                    linestyle='None',
                    alpha=alpha, fillstyle='full')
        # plt.scatter(df.loc[idx,col_names[0]], df.loc[idx,col_names[1]],
        #             c=group_colors[i], alpha=alpha, edgecolors=None)

        title += type + ":{}, ".format(np.sum(idx))
    title = title[:-2] # remove the last comma

    plt.xlabel(col_names[0])
    plt.ylabel(col_names[1])
    plt.legend(group_values, loc=loc)

    plt.title(title)

# find significantly higher or lower voxels in STA
def find_significant_voxels(sta):
    m = np.mean(sta.ravel())
    sig = np.std(sta.ravel())

    voxel_high = ((sta - m) > 2.58 * sig).astype(int)
    voxel_low = ((sta - m) < -2.58 * sig).astype(int)

    return voxel_high, voxel_low


def count_significant_voxels(sta):
    voxel_high, voxel_low = find_significant_voxels(sta)

    return np.sum(voxel_high.ravel()), np.sum(voxel_low.ravel())


# find significantly higher or lower voxels in STA slice
def find_significant_pixels(sta, time_bin, target_shape=None):
    m = np.mean(sta.ravel())
    sig = np.std(sta.ravel())

    pixel_high = ((sta[:, time_bin] - m) > 2.58 * sig).astype(int)
    pixel_low = ((sta[:, time_bin] - m) < -2.58 * sig).astype(int)

    if target_shape is not None:
        pixel_high = pixel_high.reshape(target_shape)
        pixel_low = pixel_low.reshape(target_shape)


    return pixel_high, pixel_low


def count_significant_pixels(sta, time_bin):
    pixel_high, pixel_low = find_significant_pixels(sta, time_bin)
    return np.sum(pixel_high), np.sum(pixel_low)


def calc_peak_to_peak_and_std(stim):
    stim_max = np.max(stim.ravel())
    stim_min = np.min(stim.ravel())
    stim_std = np.std(stim.ravel())

    return stim_max-stim_min, stim_std


def calc_PSNR(stim):
    stim_max = np.max(stim.ravel())
    stim_min = np.min(stim.ravel())
    stim_std = np.std(stim.ravel())

    return (stim_max-stim_min) / stim_std

# helper functions for quantifying RF
# find weighted center of significant pixels
def calc_mean_and_cov(X, weight):

    m = np.average(X, weights=weight, axis=0)

    if X.shape[0] > 1: 
        C = np.cov(X, rowvar=False, aweights=weight)
        #np.cov(X, rowvar=False, aweights=weight, ddof=0)
    else: # single pixel
        C = np.zeros((X.shape[1],X.shape[1]))
    return m, C

# # test code
# X = np.array([[1], [2], [3], [4]])
# weight = np.array([0.15, 0.35, 0.25, 0.25])
# print(X.shape)
# print(weight.shape)

# print(calc_mean_and_cov(X, weight))

# print(np.cov(X, rowvar=False))
# print(np.cov(X, rowvar=False, ddof=0))
# print(np.cov(X, rowvar=False, aweights=weight))
# print(np.cov(X, rowvar=False, aweights=weight, ddof=0))


def calc_center_and_cov(weight, mask):

    height, width = mask.shape

    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    xx, yy = np.meshgrid(x, y)

    XY = np.column_stack([xx.ravel(), yy.ravel()])
    weight = weight.ravel()

    # choose rows with non-zero mask
    if np.sum(mask.ravel()) == 0:
        return None, None
    idx = mask.ravel() > 0
    XY = XY[idx,:]
    weight = weight[idx]

    return calc_mean_and_cov(XY, weight / np.sum(weight))


# put into one function
def plot_RF(sta, time_bin, shape, plot_sta_slice=True):
    #subtract mean
    sta_mean = np.mean(sta.ravel())

    if plot_sta_slice:
        plt.imshow(sta[:,time_bin].reshape(shape), cmap='gray', origin='lower')

    # find significant pixels
    pixel_high, pixel_low = find_significant_pixels(sta, time_bin, shape)

    num_pixel_high = np.sum(pixel_high.ravel())
    num_pixel_low = np.sum(pixel_low.ravel())
    # print(num_pixel_high, num_pixel_low)

    if num_pixel_high == 0 and num_pixel_low ==0: # no significant pixel
        return None

    # find center and cov of RFs
    if num_pixel_high > 0:
        high_center, high_cov = calc_center_and_cov(sta[:,time_bin]-sta_mean, pixel_high)
    else:
        high_center = None

    if num_pixel_low > 0:
        low_center, low_cov = calc_center_and_cov(sta[:,time_bin]-sta_mean, pixel_low)
    else:
        low_center = None

    # plot as ellipse
    if high_center is not None:
        plot_ellipse(high_center, high_cov, 'r--')

        RF = {"type": "ON", "center": high_center, "cov": high_cov}

    if low_center is not None:
        plot_ellipse(low_center, low_cov, 'b--')

        RF = {"type": "OFF", "center": low_center, "cov": low_cov}

    return RF


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


def plot_ellipse_contour(width, height, mean, sig, color='r', ax=None):

    def pdf_standard_norm(z, dim=1):
        if dim == 1:
            return np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
        elif dim == 2:
            return np.exp(-z ** 2) / (2 * np.pi)

    xx, yy = np.mgrid[0:width-1:.1, 0:height-1:.1]
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


# plot histogram with the order that I WANT
# https://stackoverflow.com/questions/28418988/how-to-make-a-histogram-from-a-list-of-strings-in-python
def plot_hist(data, types=["ON", "OFF", "unknown"], width=0.3):
    from collections import Counter
    type_counts = Counter(data)

    type_counts = [type_counts[type] for type in types]
    # print(type_counts)

    plt.bar(types, type_counts, width=width)
    # plt.ylim(0,25)
    plt.ylabel('count')
    # plt.xlabel('cell type')

    return type_counts