import numpy as np
import matplotlib.pyplot as plt


# helper functions for quantifying RF
# find weighted center of significant pixels
def calc_mean_and_cov(X, weight):
    m = np.average(X, weights=weight, axis=0)

    if X.shape[0] > 1:
        cov = np.cov(X, rowvar=False, aweights=weight)
        #np.cov(X, rowvar=False, aweights=weight, ddof=0)
    else: # single pixel
        # cov = np.zeros((X.shape[1],X.shape[1]))
        cov = np.diag([1/12,1/12])

    # check degenerate case
    if np.sum(np.array(cov)>0) == 1:
        diag = np.diagonal(cov)
        if diag[0] == 0:
            cov[0,0] = 1/12
        else:
            cov[1,1] = 1/12

    return m, cov


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


def plot_ellipse(avg, covariance, color='r', linestyle='--', ax=None):

    # if np.max(covariance.ravel())==0 and np.min(covariance.ravel())==0: # single pixel
    #     plt.plot(avg[0], avg[1], 'o'+LINE_TYPE)
    #     return

    theta = np.linspace(0, 2*np.pi, 100).ravel()

    circ = np.column_stack([np.cos(theta), np.sin(theta)])

    eps = 1e-9
    L = np.linalg.cholesky(covariance+np.diag([eps, eps]))
    scale_factor = np.sqrt(6) # to match covariance of 12 to sqrt(2)/2
    ellipse = avg + circ @ L.T @ np.diag([scale_factor, scale_factor])
    # ellipse = avg + circ @ L.T @ np.diag([2.4477, 2.4477])

    if ax is None:
        plt.plot(ellipse[:,0], ellipse[:,1], color=color, linestyle=linestyle)
    else:
        ax.plot(ellipse[:, 0], ellipse[:, 1], color=color, linestyle=linestyle)

    return

def calc_axes(cov):
    eig_val = np.linalg.eig(cov)[0]
    eig_val = np.sort(eig_val)[::-1]

    axes = np.sqrt(eig_val) * np.sqrt(6)
    return axes

# plot function
def plot_sta_slice_with_receptive_field(sta, time_bin=5, target_shape=None, type='both', vmax=None, vmin=None):
    if sta.ndim == 3:
        sta_slice = sta[:,:,time_bin]
        # target_shape = sta.shape[:2]
    elif sta.ndim == 2:   # for backward compatibility
        sta_slice = sta[:, time_bin].reshape(target_shape)
        if target_shape is None:
            raise ValueError('Must provide target_shape')

    plt.imshow(sta_slice, cmap='gray', origin='lower', vmax=vmax, vmin=vmin)

    RF = fit_receptive_field(sta, time_bin, target_shape, type=type)

    if type is 'both':
        if RF[0] is not None:
            plot_receptive_field(RF[0])
        if RF[1] is not None:
            plot_receptive_field(RF[1])
    else:
        if RF is not None:
            plot_receptive_field(RF)

    return RF


def fit_receptive_field(sta, time_bin, target_shape=None, type='both'):
    #subtract mean
    if type == 'both':
        return fit_receptive_field(sta, time_bin, target_shape=target_shape, type='ON'), fit_receptive_field(sta, time_bin, target_shape=target_shape, type='OFF')

    # find global mean and std of STA
    sta_mean = np.mean(sta.ravel())
    sta_sig = np.std(sta.ravel())

    if sta.ndim == 3:
        sta_slice = sta[:,:,time_bin]
    elif sta.ndim == 2:
        sta_slice = sta[:, time_bin]

    # if type is 'both':
    significant_pixel = find_significant_pixels(sta_slice, sta_mean, sta_sig, target_shape, type)

    num_significant_pixel = np.sum(significant_pixel.ravel())

    if num_significant_pixel == 0: # no significant pixel
        return None

    # find center and cov of RFs
    rf_center, rf_cov = calc_center_and_cov(sta_slice-sta_mean, significant_pixel)
    return {"type": type, "center": rf_center, "cov": rf_cov, 'num_significant_pixels': num_significant_pixel}


def choose_opposite_RFs(RF0, RF1, ordered=True):
    check0 = [RF0[0] is None, RF0[1] is None]
    check1 = [RF1[0] is None, RF1[1] is None]

    check0 = np.where(check0)[0]
    check1 = np.where(check1)[0]
    # print(check0, check1)

    if len(check0) == 0:
        idx1 = 1 - check1[0]
        idx0 = 1 - idx1
    else:
        idx0 = 1 - check0[0]
        idx1 = 1 - idx0

    # order RFs by ON and OFF
    if ordered:
        if RF0[idx0]['type'] == 'ON':
            return RF0[idx0], RF1[idx1]
        else:
            return RF1[idx1], RF0[idx0]
    else:
        return RF0[idx0], RF1[idx1]


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


# find significantly higher or lower pixels in STA slice
def find_significant_pixels(sta, m, sig, target_shape=None, type='both'):

    if type == 'both':
        return find_significant_pixels(sta, m, sig, target_shape=target_shape, type='ON'), find_significant_pixels(sta, m, sig, target_shape=target_shape, type='OFF'),

    if type == 'ON':
        significant_pixel = ((sta - m) > 2.58 * sig).astype(int)
    else: # type is 'OFF':
        significant_pixel = ((sta - m) < -2.58 * sig).astype(int)

    if target_shape is None:
        significant_pixel = significant_pixel.reshape(sta.shape)
    else:
        significant_pixel = significant_pixel.reshape(target_shape)

    return significant_pixel


# def find_significant_pixels(sta, m, sig, target_shape=None):
#
#     pixel_high = ((sta - m) > 2.58 * sig).astype(int)
#     pixel_low = ((sta - m) < -2.58 * sig).astype(int)
#
#     if target_shape is None:
#         pixel_high = pixel_high.reshape(sta.shape)
#         pixel_low = pixel_low.reshape(sta.shape)
#     else:
#         pixel_high = pixel_high.reshape(target_shape)
#         pixel_low = pixel_low.reshape(target_shape)
#
#     return pixel_high, pixel_low


def count_significant_pixels(sta, time_bin):
    pixel_high, pixel_low = find_significant_pixels(sta, time_bin)
    return np.sum(pixel_high), np.sum(pixel_low)


def plot_receptive_field(RF, channel_name=None, linestyle='-', ax=None):
    if RF is None:
        return

    if 'type' in RF:
        if RF['type'] == 'ON':
            plot_ellipse(RF['center'], RF['cov'], color='r', linestyle=linestyle, ax=ax)
        elif RF['type'] == 'OFF':
            plot_ellipse(RF['center'], RF['cov'], color='b', linestyle=linestyle, ax=ax)
    else:
        plot_ellipse(RF['center'], RF['cov'], color='k', linestyle=linestyle, ax=ax)

    if channel_name is not None:
        text_offset_x = -0.15
        text_offset_random = 0.05
        if ax is not None:
            ax.text(RF['center'][0] + text_offset_random * np.random.randn() + text_offset_x,
                    RF['center'][1] + text_offset_random * np.random.randn(),
                    channel_name)
        else:
            plt.text(RF['center'][0] + text_offset_random * np.random.randn() + text_offset_x,
                     RF['center'][1] + text_offset_random * np.random.randn(),
                     channel_name)

#plot_sta_slice=True
