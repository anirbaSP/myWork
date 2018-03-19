import os
from os import listdir
from os.path import isfile, join

import numpy as np

# import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp
from PIL import Image
import random
from itertools import combinations

import isx
import isxrgb
import myplotmodules as mpm
import mymodels as mm

analyze_example_frame = True
analyze_time_pixel = False
analyze_time = False
correct_bad_pixels = True
correct_stray_light = False
clim = (0, 2000)

example_frame_idx = [100]
n_select_time_points = 50
n_select_pixels = 20
random_pixels = False
measure_name = 'rgb_ratio'
fit_method = 'linear'  #_zero_intercept'  #'linear'  # None #'exp'


# always initialize the API before use
def main():
    isx.initialize()

    # input files
    root_dir = ('/ariel/data2/Alice/NV3_DualColor/D_Lab/' #NV3_color_sensor_12bit/'    #
                'Masa/20170816/led2') #'/Scope_Autofluorescence/NV3-04/led2_0_gain1_coverOff')  #'V3-63/20170807/led2')  #'V3-17/20170714') #'V3-55/20170710') #'V3-55/V3_55_20170717') #'V3-75/20171127/led2') #'V3_39/20170807/led2')  #'V3-17/20170714')  #'Scope_Autofluorescence/NV3-04/led2_4_gain1_coverOff')  #NV3-04/led2_4_coverOff')  #
    # 'Scope_Autofluorescence/NV3-01/led2_0') #
    # root_dir = '/Users/Sabrina/workspace/data/pipeline_s'
    fn = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
    print('{} files have been found, they are \n {}'.format(len(fn), fn))

    # find the rgb channel, and sort fn as r/g/b
    ch = ['red', 'green', 'blue']
    rgb_files, rgb_files_root = isxrgb.find_rgb_channels(fn, channel_list=ch)
    print('rgb files are \n {}'.format(rgb_files))

    exp = isxrgb.get_exp_label(rgb_files_root)

    rgb_files_with_path = [os.path.join(root_dir, file) for file in rgb_files]

    # open one channel first to get the basics of the movie
    ext = os.path.splitext(rgb_files_with_path[0])[1]

    if ext == '.isxd':
        tmp = isx.Movie(rgb_files_with_path[0])
        frame_shape = tmp.shape
        n_pixels = frame_shape[0] * frame_shape[1]
        n_frames = tmp.num_frames
        frame_rate = tmp.frame_rate
        tmp.close()
    elif ext == '.tif':
        tmp = Image.open(rgb_files_with_path[0])
        frame_shape = tmp.size[::-1]
        n_pixels = frame_shape[0] * frame_shape[1]
        n_frames = tmp.n_frames
        frame_rate = 20 #tmp.frame_rate #todo: does the tif file always have frame rate info? what's the tag name??
        tmp.close()

    # get an example frame to get accurate frame_shape (especially necessary when correct_bad_pixels == True
    tmp = isxrgb.get_rgb_frame(rgb_files_with_path, 0, correct_stray_light=correct_stray_light,
                               correct_bad_pixels=correct_bad_pixels)
    frame_shape = tmp.shape[1:3]
    n_pixels = frame_shape[0] * frame_shape[1]

    ####
    if analyze_example_frame:
        for frame_idx in example_frame_idx:
            # get data
            example_rgb_frame = isxrgb.get_rgb_frame(rgb_files_with_path, frame_idx,
                                                     camera_bias=None,
                                                     correct_stray_light=correct_stray_light,
                                                     correct_bad_pixels=correct_bad_pixels)
            # prepare figure
            fig = plt.figure(figsize=(18, 8))
            gs = gsp.GridSpec(2, 4, wspace=0.2)
            plt.suptitle('{}\n\n{}'.format(exp['label'], rgb_files_root)) #, rgb_files_root, frame_idx))   #{}  frame # {}

            # show rgb images
            hax = list()
            for k in range(3):
                hax.append(plt.subplot(gs[0, k]))
            # tmp = np.zeros((3, 20, 30))
            # tmp[:, 0:10, 0:15] = example_rgb_frame[:, 0:10, 0:15]
            # show_rgb_frame(tmp, ax_list=hax, clim=clim)
            isxrgb.show_rgb_frame(example_rgb_frame, cmap='gray', ax_list=hax, clim=clim)

            # plot the histogram
            hax = plt.subplot(gs[0, 3])
            plot_rbg_pixels_hist(example_rgb_frame, ax=hax)
            hax.set_xlim(clim)
            pos = hax.get_position()
            pos.x0 = pos.x0 + 0.3 * pos.width
            hax.set_position(pos)

            # plot scatter plot to show the ratio for r / g, g / b, and b / r pair
            hax = []
            for k in range(3):
                hax.append(plt.subplot(gs[1, k]))
            show_rgb_ratio(example_rgb_frame, ax_list=hax, n_select_pixels=1000)

        gs.tight_layout(fig, rect=[0.02, 0.02, 0.96, 0.95])
        figure_name = '{}/figures/{}_file{}_frame{}'.format(os.getcwd(), exp['label'], rgb_files_root[-2:], frame_idx)
        if correct_stray_light:
            figure_name = '{}_correct'.format(figure_name)
        plt.savefig(figure_name, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0,
                    frameon=None)

    ####
    if analyze_time_pixel:
        # get data
        select_frame_idx = np.arange(0, n_frames - 1, int(n_frames / n_select_time_points))
        select_pixel_idx = np.arange(0, n_pixels - 1, int(n_pixels / n_select_pixels))
        if random_pixels:
            select_pixel_idx = random.sample(range(n_pixels), int(n_pixels / n_select_pixels))

        rgb_pixel_time = isxrgb.get_rgb_pixel_time(rgb_files_with_path, select_frame_idx, select_pixel_idx,
                                                   correct_stray_light=correct_stray_light,
                                                   correct_bad_pixels=correct_bad_pixels)
        # make figure for plots
        plt.figure(figsize=(10, 8))
        gs = gsp.GridSpec(3, 2, width_ratios=[3, 1], wspace=0.3)
        plt.suptitle('{}\n\n {}  selected {} frames'.format(exp['label'], rgb_files_root, len(select_frame_idx)))

        hax = list()
        for i in range(2):
            for k in range(3):
                hax.append(plt.subplot(gs[k, i]))  # for k in range(3) and i in range(2)))

        plot_rgb_pixel_time(rgb_pixel_time, select_frame_idx, frame_rate, measure_name=measure_name, ax_list=hax,
                            fit_method=fit_method, ch=None)

        # plt.show()
        figure_name = '{}/figures/{}_time_{}_file{}'.format(os.getcwd(), exp['label'], measure_name, rgb_files_root[-2:])
        if correct_stray_light:
            figure_name = '{}_correct'.format(figure_name)
        plt.savefig(figure_name, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None)

    if analyze_time:
        select_frame_idx = np.arange(0, n_frames - 1, int(n_frames / n_select_time_points))
        rgb_stack = np.empty((len(ch), len(select_frame_idx)))
        print('Collect frame', end='')
        for i, frameIdx in enumerate(select_frame_idx):  # show randomly selected frames
            print('...', end='')
            this_rgb_frame = isxrgb.get_rgb_frame(rgb_files_with_path, frameIdx,
                                                  correct_stray_light=correct_stray_light,
                                                  correct_bad_pixels=correct_bad_pixels)
            # calculate average intensity across the frame for R/G/B
            rgb_stack[:, i] = np.mean(this_rgb_frame.reshape(-1, len(ch)), axis=0)
            print('frame {}'.format(frameIdx))

        # total_drop = (rgb_stack[:, 0] - rgb_stack[:, -1])/rgb_stack[:, 0]

        plt.figure(figsize=(10, 10))
        gs = gsp.GridSpec(3, 1)
        htitle = plt.suptitle(
            '{}\n\n {} photobleaching selected {} frames'.format(exp['label'], rgb_files_root, len(select_frame_idx)))

        hax = []
        for k in range(len(ch)):
            hax.append(plt.subplot(gs[k]))

        for i in range(len(ch)):
            plt.sca(hax[i])
            x2plot = select_frame_idx / frame_rate / 60
            y2plot = rgb_stack[i, :]
            plt.plot(x2plot, y2plot, color=ch[i], linestyle='None', marker='.', markersize=2)

            a, k, c, y_hat = mm.fit_exponential(x2plot, y2plot, linear_method=True, c0=0)
            a, k, c, y_hat = mm.fit_exponential(x2plot, y2plot, linear_method=False, p0=[a, k, c])
            plt.plot(x2plot, y_hat, label='$F = %0.2f e^{%0.2f t} + %0.2f$' % (a, k, c))
            hax[i].legend(bbox_to_anchor=(1, 1))
            hax[i].set_ylabel('F_mean', color=ch[i])
            pb_measured = np.diff(y2plot[[-1, 0]]) / y2plot[0]
            pb_fitted = np.diff(y_hat[[-1, 0]]) / y2plot[0]
            pb_measured_s = '{:.0%}'.format(pb_measured[0])
            pb_fitted_s = '{:.0%}'.format(pb_fitted[0])
            label = '{}           {}'.format(pb_measured_s, pb_fitted_s)

            if i == 0:
                label = 'Total fluorescence drop in {:.1f} min is measured as {} fitted as {}'.format(x2plot[-1],
                                                                                                      pb_measured_s,
                                                                                                      pb_fitted_s)
            plt.text(x2plot[-1], hax[i].get_ylim()[-1] - 0.2 * np.diff(hax[i].get_ylim()), label,
                     horizontalalignment='right', verticalalignment='top', color=ch[i])
        hax[-1].set_xlabel('time (min)')

        # plt.show()
        figure_name = '{}/figures/{}_photobleaching_file{}'.format(os.getcwd(), exp['label'], rgb_files_root[-2:])
        if correct_stray_light:
            figure_name = '{}_correct'.format(figure_name)
        plt.savefig(figure_name, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None)


def show_rgb_subtracted_frame():  #to be done if necessary
    pass
    # plot the subtracted blue image from green
    # hax = plt.subplot(gs[1, 2])
    # select_ch = ['blue', 'green']
    # find = lambda search_list, elem: [[i for i, x in enumerate(search_list) if x == e] for e in elem]
    # select_ch_idx = find(['red', 'green', 'blue'], select_ch)
    # frame2plot = example_rgb_frame[:, :, select_ch_idx[0][0]] - example_rgb_frame[:, :, select_ch_idx[1][0]]
    # show_frame(frame2plot, ax=hax)
    # plt.title('{} - {}'.format(select_ch[0], select_ch[1]))


def plot_rgb_pixels_dist(rgb_frame, sort_channel, ax=None, ax1=None):
    ch = ['red', 'green', 'blue']
    sort_channel_idx = ch.index(sort_channel)
    sort_channel_pixels = rgb_frame[:, :, sort_channel_idx]
    pix_idx = np.argsort(sort_channel_pixels.flatten())

    # (row_num, col_num, ch_num) = rgb_frame.shape
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])

    plt.sca(ax)
    hp = list()
    for i in range(len(ch)):
        y2plot = rgb_frame[:, :, i].flatten()
        hp0 = plt.plot(y2plot[pix_idx], color=ch[i], alpha=0.4)
        hp.append(hp0)

    # hp[2][0].set_alpha(0.5)  # Blue is too strong and cover green, so set transpancy for blue
    # hp[1][0].set_zorder(10)  # Bring green to the top

    plt.xlabel('pixels (sorted)')
    plt.ylabel('Intensity')
    plt.title('Pixel-wise analysis')

    if ax1 is not None:
        # plot normalized pixel value for all channels, normalized to sort_channel
        # pixel-based normalization
        tmp = rgb_frame[:, :, sort_channel_idx]
        rgb_frame_norm = rgb_frame / tmp[:, :, np.newaxis]
        plt.sca(ax1)
        for i in range(len(ch)):
            y2plot = rgb_frame_norm[:, :, i].flatten()
            hp0 = plt.plot(y2plot[pix_idx], color=ch[i], alpha=0.6)
            hp.append(hp0)
        # ax1.set_ylim(0, 2)
        plt.xlabel('Pixels (sorted to {})'.format(sort_channel))
        plt.ylabel('Intensity\n normalized to {}'.format(sort_channel))
        ax.xaxis.set_label_text('')
        ax.xaxis.set_ticklabels('')


def plot_rbg_pixels_hist(rgb_frame, ax=None):
    ch = ['red', 'green', 'blue']

    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])

    for i in range(len(ch)):
        plt.sca(ax)
        n, bins, patches = plt.hist(rgb_frame[i, :, :].flatten(), 50, normed=0, facecolor=ch[i], alpha=0.4)

    for patch in patches:
        patch.set_zorder(0)
    plt.xlabel('Intensity')
    plt.ylabel('Pixel counts')


def show_frame(frame, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])

    plt.sca(ax)
    im = plt.imshow(frame, vmax=200, vmin=-200, cmap='jet', aspect='equal')
    # plt.axis_off()
    cbar = plt.colorbar()
    plt.autoscale(tight=True)


def show_rgb_ratio(rgb_frame, channel=None, ax_list=None, n_select_pixels=None, random_pixels=None):
    if channel is None:
        ch = ['red', 'green', 'blue']
    comb = combinations(range(len(ch)), 2)
    pair_list = list(comb)

    if ax_list is None:
        fig = plt.figure(figsize=(10, 3))
        gs = plt.GridSpec(1, len(pair_list))
        ax_list = [plt.subplot(gs[0, k] for k in range(3))]

    if random_pixels is None:
        random_pixels = False

    n_pixels = rgb_frame[0, :, :].size
    if n_select_pixels is None:
        n_select_pixels = n_pixels

    if random_pixels:
        select = random.sample(range(n_pixels), n_select_pixels)
    else:
        select = np.arange(0, n_pixels - 1, int(n_pixels / n_select_pixels))

    assert len(ax_list) == 3
    for i, ax in enumerate(ax_list):
        plt.sca(ax)
        x2plot = rgb_frame[pair_list[i][0], :, :].flatten()
        y2plot = rgb_frame[pair_list[i][1], :, :].flatten()
        x2plot = x2plot[select]
        y2plot = y2plot[select]
        plt.scatter(x2plot, y2plot, s=2)

        if fit_method is None:
            pass
        elif fit_method is 'linear':
            # m, b = np.polyfit(x2plot, y2plot, 1)  # fit with linear
            coeff, y_hat = mm.linear_regression(x2plot, y2plot)
            b = coeff[0]
            m = coeff[1]
        elif fit_method is 'linear_zero_intercept':
            coeff, y_hat = mm.linear_regression(x2plot, y2plot, b0=0)
            b = 0
            m = coeff

        # y_hat = m * x2plot + b
        plt.plot(x2plot, y_hat, '-r', label='$y = %0.2f x + %0.2f$' % (m, b))
        ax.legend(bbox_to_anchor=(0, 1.02), loc=3, borderaxespad=0)

        plt.axis('equal')
        plt.autoscale(tight=True)
        ax.set_xlabel(ch[pair_list[i][0]], color=ch[pair_list[i][0]])
        ax.set_ylabel(ch[pair_list[i][1]], color=ch[pair_list[i][1]])


def plot_rgb_pixel_time(rgb_pixel_time, select_frame_idx, frame_rate, measure_name=None, ax_list=None,
                        fit_method=None, ch=None):
    if ch is None:
        ch = ['red', 'green', 'blue']

    if measure_name == 'rgb_pixel':
        data2plot = rgb_pixel_time
    elif measure_name == 'rgb_ratio':
        rgb_pixel_ratio_time, pair_list = isxrgb.calc_rgb_ratio(rgb_pixel_time, ch)
        data2plot = rgb_pixel_ratio_time

    shape = data2plot.shape
    n_group = shape[0]
    n_pixel = shape[1]
    n_frame = shape[2]  # to do: check if n_frame = len(select_frame_idx)

    if ax_list is None:
        plt.figure(figsize=(10, 8))
        gs = gsp.GridSpec(n_group, 2, width_ratios=[3, 1], wspace=0.3)

        hax = list()
        for i in range(2):
            for k in range(n_group):
                hax.append(plt.subplot(gs[k, i]))

    drop = np.empty([n_group, n_pixel])
    for i in range(n_group):
        # plot time trace
        ax = ax_list[i]
        plt.sca(ax)
        x2plot = select_frame_idx / (frame_rate * 60)
        for j in range(n_pixel):
            y2plot = data2plot[i, j, :]
            plt.plot(x2plot, y2plot)

            if fit_method is None:
                drop[i, j] = (y2plot[-1] - y2plot[0]) / y2plot[0]  # estimate drop by first the last time point
            elif fit_method is 'linear':
                m, b = np.polyfit(x2plot, y2plot, 1)  # fit with linear
                yhat0 = m * x2plot[0] + b
                yhat1 = m * x2plot[-1] + b
                drop[i, j] = (yhat1 - yhat0) / yhat0
            elif fit_method is 'exp':  # fit with mono-exponential decay
                a, k, c, yhat = mm.fit_exponential(x2plot, y2plot, linear_method=True, c0=0)
                a, k, c, yhat = mm.fit_exponential(x2plot, y2plot, linear_method=False, p0=[a, k, c])
                drop[i, j] = (yhat[-1] - yhat[0]) / yhat[0]

        if i == 2:
            ax.set_xlabel('Time (min)')
        if measure_name == 'rgb_pixel':
            ax.set_ylabel(r'$F$', color=ch[i])
        elif measure_name == 'rgb_ratio':
            ax.set_ylabel('{} / {}'.format(ch[pair_list[i][0]], ch[pair_list[i][1]]))

        # plot trace drop
        ax = ax_list[i + 3]
        if measure_name == 'rgb_pixel':
            plt.sca(ax)
            plt.scatter(rgb_pixel_time[i, :, 0], drop[i, :], s=3)
            ax.set_ylabel('PB (%)', color=ch[i])  # $\Delta F/F$'
        elif measure_name == 'rgb_ratio':
            # split to two vertical axes
            hax0 = mpm.split_axes(ax, 2, 'vertical', gap=0.25)
            ax = hax0[0]
            for j in range(2):
                plt.sca(hax0[j])
                plt.scatter(rgb_pixel_time[pair_list[i][j], :, 0], drop[i, :], s=3)
                hax0[j].spines['bottom'].set_color(ch[pair_list[i][j]])
                hax0[j].tick_params(axis='x', colors=ch[pair_list[i][j]])
                # plt.axhline(linewidth=0.5, color='black')
                if i == n_group - 1 and j == 0:
                    ax.set_ylabel('Slope')
        if i == n_group - 1:
            ax.set_xlabel(r'$F_0$')


if __name__ == '__main__': main()
