import isx
import isxrgb

import os
from os import listdir
from os.path import isfile, join

import numpy as np
from PIL import Image
# import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp

from itertools import combinations
import random

correct_stray_light = False
correct_bad_pixels = True
n_select_pixels = 1000
random_pixels = False
time_range = [0, 0.03]  # second

root_dir_group_list = [['/ariel/data2/Alice/NV3_DualColor/NV3_color_sensor_12bit/Scope_Autofluorescence/NV3-01',
                        'led1_6', 'led1_5', 'led1_4', 'led1_3', 'led1_2', 'led1_1', 'led1_0']]

# ['/Volumes/data2/Alice/NV3_DualColor/NV3_color_sensor_12bit/V3_71/20171127', 'led2'],
# ['/Volumes/data2/Alice/NV3_DualColor/NV3_color_sensor_12bit/V3_71/20171127', 'led12']
# ]
check_additivity = False


def main():

    isx.initialize()

    ch = ['red', 'green', 'blue']
    n_group = len(root_dir_group_list)
    max_n_file = max(len(this_group_root_dir_list) for this_group_root_dir_list in root_dir_group_list) - 1
    n_ch = len(ch)

    rgb_frame_file_group_info = []
    for group_idx, this_group_root_dir_list in enumerate(root_dir_group_list):
        rgb_frame_file_group_info.append({})
        rgb_frame_file_group_info[group_idx]['tissue'] = []
        rgb_frame_file_group_info[group_idx]['microscope'] = []
        rgb_frame_file_group_info[group_idx]['led_name'] = []
        rgb_frame_file_group_info[group_idx]['led_power'] = []

        for file_idx in range(len(this_group_root_dir_list) - 1):
            root_dir = os.path.join(this_group_root_dir_list[0], this_group_root_dir_list[file_idx + 1])
            fn = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
            print('{} files have been found, they are \n {}'.format(len(fn), fn))

            # find the rgb channel, and sort fn as r/g/b
            rgb_filenames, rgb_basename = isxrgb.find_rgb_channels(fn)
            print('rgb files are \n {}'.format(rgb_filenames))

            exp_label = isxrgb.get_exp_label(rgb_basename)

            rgb_filenames_with_path = [os.path.join(root_dir, file) for file in rgb_filenames]

            header = isxrgb.MovieHeader(rgb_filenames_with_path[0])
            if correct_bad_pixels:
                header.correct_bad_pixels()

            frame_range = np.array(time_range) * header.frame_rate
            select_frame_idx = np.arange(frame_range[0], frame_range[1])
            rgb_frame_stack = np.zeros((len(ch), header.n_row, header.n_col, len(select_frame_idx)))
            print('Collect frame', end='')
            for i, frameIdx in enumerate(select_frame_idx):  # show randomly selected frames
                print('...', end='')
                this_rgb_frame = isxrgb.get_rgb_frame(rgb_filenames_with_path, frameIdx,
                                                      correct_stray_light=correct_stray_light,
                                                      correct_bad_pixels=correct_bad_pixels)
                rgb_frame_stack[:, :, :, i] = this_rgb_frame
                print('frame {}'.format(frameIdx))

            # calculate average intensity across time for R/G/B
            rgb_frame_stack_mean4time = np.mean(rgb_frame_stack, axis=3)
            # rgb_frame_stack_std4time = np.std(rgb_frame_stack, axis=3)

            if file_idx == 0 and group_idx == 0:
                shape = rgb_frame_stack.shape
                rgb_frame_file_group = np.empty((shape[0], shape[1], shape[2], max_n_file, n_group))

            rgb_frame_file_group[:, :, :, file_idx, group_idx] = rgb_frame_stack_mean4time

            # get experimental parameter from exp_label

            rgb_frame_file_group_info[group_idx]['tissue'].append(exp_label['tissue'])
            rgb_frame_file_group_info[group_idx]['microscope'].append(exp_label['microscope'])
            rgb_frame_file_group_info[group_idx]['led_name'].append(exp_label['led_name'])

    # select pixels to plot
    n_pixels = header.n_row * header.n_col
    if random_pixels:
        select_pixels = random.sample(range(n_pixels), n_select_pixels)
    else:
        select_pixels = np.arange(0, n_pixels - 1, int(n_pixels / n_select_pixels))

    # plot example pixels across files
    fig = plt.figure(figsize=(8, 5))  # (7, 10))
    if check_additivity:
        n_group2plot = n_group+1
    else:
        n_group2plot = n_group
    gs = plt.GridSpec(3,
                      n_group2plot,
                      left=0.05,
                      right=0.95,
                      wspace=0.3,
                      width_ratios=[4, 4, 4, 3])
    ax_list = []
    for j in range(n_group2plot):
        tmp = []
        for i in range(n_ch):
            tmp.append(plt.subplot(gs[i, j]))
        ax_list.append(tmp)
    # tmp = [ax_list[i:i + n_ch] for i in range(0, len(ax_list) - 1, n_ch)]
    # ax_list = tmp

    plot_rgb_intensity_vs_ledPower(rgb_frame_file_group,
                                   rgb_frame_file_group_info,
                                   select_pixels,
                                   ax_list=ax_list[:-1])

    for group_idx in range(n_group):
        isxrgb.show_rgb_frame(rgb_frame_file_group[:, :, :, 0, group_idx],
                              ax_list=ax_list[group_idx],
                              clim=[0, 2000],
                              cmap=None, colorbar=False)
        label = '{} ({}mW) LED'.format(rgb_frame_file_group_info[group_idx]['led_name'][0],
                                            rgb_frame_file_group_info[group_idx]['led_power'][0])
        if group_idx == 2:
            label = '{} ({}mW) LED \n+ {} ({}mW) LED'.format(
                rgb_frame_file_group_info[0]['led_name'][0], rgb_frame_file_group_info[0]['led_power'][0],
                rgb_frame_file_group_info[1]['led_name'][0], rgb_frame_file_group_info[1]['led_power'][0])
        ax_list[group_idx][0].set_title(label, fontsize=10)
        for i in [0, 1, 2]:
            ax_list[group_idx][i].axis('off')
            plt.sca(ax_list[group_idx][i])
            plt.tick_params(axis='both', which='major', labelsize=8)
        ax_list[0][0].axis('on')

    if check_additivity:
        # create predicted
        rgb_frame_predicted = np.sum(rgb_frame_file_group[:, :, :, :, [0, 1]], axis=4)
        plot_measured_vs_predicted(rgb_frame_file_group[:, :, :, :, -1], rgb_frame_predicted, select_pixels,
                                   ax_list=ax_list[:][-1])

    # camera_gain = [0, 1, 2, 3]  # todo: parse gain parameter from lookup table?
    # plot_rgb_intensity_vs_cameraGain(rgb_frame_file_group, rgb_frame_file_group_info, select_pixels, camera_gain,
    #                                  ax_list=ax_list)

    # plot_rgb_intensity_coverOn_vs_coverOff(rgb_frame_file_group, rgb_frame_file_group_info, select_pixels,
    #                                        ax_list=ax_list)

    # plt.show()
    figure_title = '{}_{}_measured_vs_predicted'.format(rgb_frame_file_group_info[group_idx]['tissue'][0],
                                                                rgb_frame_file_group_info[group_idx]['microscope'][0])
    if correct_stray_light:
        figure_title = '{}_correct'.format(figure_title)

    htitle = plt.suptitle('{}, {} example pixels'.format(figure_title, n_select_pixels))
    # if ax_list is not None:
    #     gs.tight_layout(fig, rect=[0.02, 0.02, 0.96, 0.95])

    plt.savefig('{}/figures/{}'.format(os.getcwd(), figure_title), dpi=600, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None)

    isx.shutdown()


def plot_rgb_intensity_vs_ledPower(rgb_frame_file_group, rgb_frame_file_group_info, select_pixels,
                                   ax_list=None, channel=None):

    tmp = rgb_frame_file_group.shape
    n_ch = tmp[0]  # todo check if n_ch = len(channel)
    frame_shape = tmp[1:3]
    n_files = tmp[3]
    n_group = tmp[4]
    n_pixels = frame_shape[0] * frame_shape[1]

    if channel is None:
        ch = ['red', 'green', 'blue']

    if ax_list is None:
        fig = plt.figure(figsize=(15, 10))  # (7, 10))
        gs = plt.GridSpec(3, n_group)
        ax_list = []
        for j in range(n_group):
            for i in range(n_ch):
                ax_list.append(plt.subplot(gs[i, j]))
        tmp = [ax_list[i:i + n_ch] for i in range(0, len(ax_list) - 1, n_ch)]
        ax_list = tmp

    shape0 = (n_ch, frame_shape[0] * frame_shape[1], n_files, n_group)

    # plot example pixels for each group
    for group_idx in range(min(len(root_dir_group_list) + 1, len(ax_list))):
        for ch_idx in range(len(ch)):
            ax = ax_list[group_idx][ch_idx]
            plt.sca(ax)
            x2plot = rgb_frame_file_group_info[group_idx]['led_power']
            for pixel_idx in select_pixels:
                plt.plot(x2plot, rgb_frame_file_group.reshape(shape0)[ch_idx, pixel_idx, :, group_idx])

            if group_idx == 0:
                plt.ylabel(ch[ch_idx], color=ch[ch_idx])
            ax.set_xticks(x2plot)

            if ch_idx == 2:
                if group_idx < n_group:
                    xlabel_string = '{} LED'.format(rgb_frame_file_group_info[group_idx]['led_name'][0])
                elif group_idx == 2:
                    xlabel_string = 'Blue LED + Lime LED'
                else:
                    xlabel_string = 'Predicted by addition'
                plt.xlabel(xlabel_string)

    # plt.show()
    figure_title = '{}_{}_intensity_vs_ledPower_byPixel'.format(rgb_frame_file_group_info[group_idx]['tissue'][0],
                                                     rgb_frame_file_group_info[group_idx]['microscope'][0])
    if correct_stray_light:
        figure_title = '{}_correct'.format(figure_title)

    htitle = plt.suptitle('{}, {} example pixels'.format(figure_title, n_select_pixels))


def plot_measured_vs_predicted(rgb_frame_file_measured, rgb_frame_file_predicted, select_pixels, ax_list=None,
                               channel=None):
    if channel is None:
        ch = ['red', 'green', 'blue']

    tmp = rgb_frame_file_measured.shape
    n_ch = tmp[0]  # todo check if n_ch = len(channel)
    frame_shape = tmp[1:3]
    n_files = tmp[3]
    n_pixels = frame_shape[0] * frame_shape[1]

    if ax_list is None:
        fig = plt.figure(figsize=(5, 10))
        gs = plt.GridSpec(n_ch, width_ratios=[4, 4, 4, 2])
        # ax_list = []
        ax_list.append(plt.subplot(gs[i]) for i in range(n_ch))

    # plot scatter plot predicted vs measured
    for ch_idx in range(n_ch):
        ax = ax_list[ch_idx]
        plt.sca(ax)
        plt.scatter(rgb_frame_file_measured.reshape([n_ch, n_pixels, -1])[ch_idx, select_pixels, :],
                    rgb_frame_file_predicted.reshape([n_ch, n_pixels, -1])[ch_idx, select_pixels, :], s=1)
        plt.autoscale(tight=True)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        # plt.axis('equal')
        plt.plot(ax.get_xlim(), ax.get_ylim(), 'k-', lw=0.5)
        plt.tick_params(axis='both', which='major', labelsize=8)
        ax.set_ylabel('Predicted', color=ch[ch_idx])
        if ch_idx == 2:
            ax.set_xlabel('Measured')

def plot_rgb_intensity_vs_cameraGain(rgb_frame_file_group, rgb_frame_file_group_info, select_pixels, camera_gain,
                                     ax_list=None, channel=None):

    tmp = rgb_frame_file_group.shape
    n_ch = tmp[0]  # todo check if n_ch = len(channel)
    frame_shape = tmp[1:3]
    n_files = tmp[3]
    n_group = tmp[4]
    n_pixels = frame_shape[0] * frame_shape[1]

    if channel is None:
        ch = ['red', 'green', 'blue']

    if ax_list is None:
        fig = plt.figure(figsize=(7, 10))
        gs = plt.GridSpec(n_ch, n_group)
        ax_list = []
        for j in range(n_group):
            for i in range(n_ch):
                ax_list.append(plt.subplot(gs[i, j]))
        tmp = [ax_list[i:i + n_ch] for i in range(0, len(ax_list) - 1, n_ch)]
        ax_list = tmp

    shape0 = (len(ch), frame_shape[0] * frame_shape[1], n_files, n_group)

    # plot example pixels for each group
    for group_idx in range(min(len(root_dir_group_list) + 1, len(ax_list))):
        for ch_idx in range(len(ch)):
            ax = ax_list[group_idx][ch_idx]
            plt.sca(ax)
            # x2plot = rgb_frame_file_group_info[group_idx]['led_power']
            x2plot = camera_gain
            for pixel_idx in select_pixels:
                plt.plot(x2plot, rgb_frame_file_group.reshape(shape0)[ch_idx, pixel_idx, :, group_idx])

            if group_idx == 0:
                plt.ylabel(ch[ch_idx], color=ch[ch_idx])
            ax.set_xticks(x2plot)

            if ch_idx == 2:
                xlabel_string = 'Camera gain'
                plt.xlabel(xlabel_string)

            if ch_idx == 0:
                plt.title('{} LED {}mW'.format(rgb_frame_file_group_info[group_idx]['led_name'][ch_idx],
                                               rgb_frame_file_group_info[group_idx]['led_power'][ch_idx]))

    figure_title = '{}_{}_CameraGain_byPixel'.format(rgb_frame_file_group_info[group_idx]['tissue'][0],
                                                     rgb_frame_file_group_info[group_idx]['microscope'][0])

    if correct_stray_light:
        figure_title = '{}_correct'.format(figure_title)

    htitle = plt.suptitle('{}, {} example pixels'.format(figure_title, n_select_pixels))


def plot_rgb_intensity_coverOn_vs_coverOff(rgb_frame_file_group, rgb_frame_file_group_info, select_pixels,
                                   ax_list=None, channel=None):

    tmp = rgb_frame_file_group.shape
    n_ch = tmp[0]  # todo check if n_ch = len(channel)
    frame_shape = tmp[1:3]
    n_files = tmp[3]
    n_group = tmp[4]
    n_pixels = frame_shape[0] * frame_shape[1]

    if channel is None:
        ch = ['red', 'green', 'blue']

    if ax_list is None:
        fig = plt.figure(figsize=(15, 10))  # (7, 10))
        gs = plt.GridSpec(n_ch, n_group)
        ax_list = []
        for j in range(n_group):
            for i in range(n_ch):
                ax_list.append(plt.subplot(gs[i, j]))
        tmp = [ax_list[i:i + n_ch] for i in range(0, len(ax_list) - 1, n_ch)]
        ax_list = tmp

    shape0 = (n_ch, frame_shape[0] * frame_shape[1], n_files, n_group)

    # plot example pixels for each group
    for group_idx in range(min(len(root_dir_group_list) + 1, len(ax_list))):
        for ch_idx in range(len(ch)):
            ax = ax_list[group_idx][ch_idx]
            plt.sca(ax)
            xy2plot = rgb_frame_file_group.reshape(shape0)[ch_idx, select_pixels, :, group_idx]
            plt.scatter(xy2plot[:, 0], xy2plot[:, 1], s=2)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            lim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
            ax.set_xlim(lim)
            ax.set_ylim(lim)

            plt.plot(ax.get_xlim(), ax.get_ylim(), 'k-', lw=0.5)

            if group_idx == 0:
                ax.set_ylabel('Cover OFF', color=ch[ch_idx])
                if ch_idx == 2:
                    ax.set_xlabel('Cover ON', color='black')
            elif group_idx == n_group-1:
                ax.set_ylabel('Cover ON_1', color=ch[ch_idx])
                if ch_idx == 2:
                    ax.set_xlabel('Cover ON_2', color='black')

            if ch_idx == 0:
                if group_idx == n_group - 1:
                    ax.set_title('{} LED {}mW\nOLD vs NEW'.format(rgb_frame_file_group_info[group_idx]['led_name'][ch_idx],
                                                      rgb_frame_file_group_info[group_idx]['led_power'][ch_idx]))
                else:
                    ax.set_title('{} LED {}mW'.format(rgb_frame_file_group_info[group_idx]['led_name'][ch_idx],
                                               rgb_frame_file_group_info[group_idx]['led_power'][ch_idx]))



    figure_title = '{}_{}_Cover_byPixel'.format(rgb_frame_file_group_info[group_idx]['tissue'][0],
                                                     rgb_frame_file_group_info[group_idx]['microscope'][0])
    if correct_stray_light:
        figure_title = '{}_correct'.format(figure_title)

    htitle = plt.suptitle('{}, {} example pixels'.format(figure_title, n_select_pixels))


if __name__ == '__main__': main()
