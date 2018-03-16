from __future__ import division

import isx
import isxrgb
import myutilities as mu
import myplotmodules as mp

import os
from os import listdir
from os.path import isfile, join

import sys

import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import json

correct_stray_light = False
correct_bad_pixels = True
n_select_pixels = 1000
random_pixels = False
time_range = [0, 1]  # second
max_n_frame = 100

# root_dir_group_list = [['/Volumes/data2/Alice/NV3_DualColor/NV3_color_sensor_12bit/V3-17', '20170717', 'tmp'],
#                        ['/Volumes/data2/Alice/NV3_DualColor/NV3_color_sensor_12bit/V3-17', '20170714', 'tmp']
#                        ]
# root_dir_group_list = [['/Volumes/data2/Alice/NV3_DualColor/NV3_color_sensor_12bit/V3-55', 'V3_55_20170717'],
#                        ['/Volumes/data2/Alice/NV3_DualColor/NV3_color_sensor_12bit/V3-55', '20170710']
#                        ]
# root_dir_group_list = [['/Volumes/data2/Alice/NV3_DualColor/NV3_color_sensor_12bit/V3-63/20170807', 'led1'],
#                        ['/Volumes/data2/Alice/NV3_DualColor/NV3_color_sensor_12bit/V3-63/20170807', 'led2']
#                        ]
root_dir_group_list = [['/Volumes/data2/Sabrina/data/dualcolor/GCaMP/NV3-01_V3-17_20170717', 'led1'],
                       ['/Volumes/data2/Sabrina/data/dualcolor/GCaMP/NV3-01_V3-17_20170714', 'led2']
                       ]


def main():
    file_index = 0
    # current only handle the first file in each group. todo: for multiple files
    for i in range(len(root_dir_group_list)):
        root_dir_group_list[i] = root_dir_group_list[i][0:2]

    # isx.initialize()
    ch = ['red', 'green', 'blue']
    n_group = len(root_dir_group_list)
    max_n_file = max(len(this_group_root_dir_list) for this_group_root_dir_list in root_dir_group_list) - 1
    n_ch = len(ch)

    rgb_frame_file_group_info = []
    cur_short_recording = math.inf
    """
        get file info first, if there is any input file info unexpected, we can error ahead of time
    """
    for group_idx, this_group_root_dir_list in enumerate(root_dir_group_list):
        rgb_frame_file_group_info.append({})
        rgb_frame_file_group_info[group_idx]['tissue'] = []
        rgb_frame_file_group_info[group_idx]['microscope'] = []
        rgb_frame_file_group_info[group_idx]['led_name'] = []
        rgb_frame_file_group_info[group_idx]['led_power'] = []
        rgb_frame_file_group_info[group_idx]['rgb_file_root_with_path'] = []
        for file_idx in range(len(this_group_root_dir_list) - 1):
            root_dir = os.path.join(this_group_root_dir_list[0], this_group_root_dir_list[file_idx + 1])
            fn = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
            print('{} files have been found, they are \n {}'.format(len(fn), fn))

            # find the rgb channel, and sort fn as r/g/b
            rgb_files, rgb_files_root = isxrgb.find_rgb_channels(fn)
            print('rgb files are \n {}'.format(rgb_files))

            """
                get experiment info
            """
            exp = isxrgb.get_exp_label(rgb_files_root)
            rgb_frame_file_group_info[group_idx]['led_power'].append(exp['led_power'])
            rgb_frame_file_group_info[group_idx]['tissue'].append(exp['tissue'])
            rgb_frame_file_group_info[group_idx]['microscope'].append(exp['microscope'])
            rgb_frame_file_group_info[group_idx]['led_name'].append(exp['led_name'])
            rgb_frame_file_group_info[group_idx]['rgb_file_root_with_path'] = os.path.join(root_dir, rgb_files_root)

            """
                Open each files and validate conditions before massive reading frames in the next step.
                This step prepare for the massive file reading and stop the code if input files are not valid.
            """
            rgb_files_with_path = [os.path.join(root_dir, file) for file in rgb_files]

            # open one channel first to get the frame numbers
            ext = os.path.splitext(rgb_files_with_path[0])[1]

            if ext == '.isxd':
                tmp = isx.Movie(rgb_files_with_path[0])
                n_frames = tmp.num_frames
                frame_rate = tmp.frame_rate
                tmp.close()
            elif ext == '.tif':
                tmp = Image.open(rgb_files_with_path[0])
                n_frames = tmp.n_frames
                frame_rate = tmp.frame_rate
                tmp.close()

            frame_range = np.array(time_range) * frame_rate
            cur_short_recording = min(cur_short_recording, n_frames/frame_rate)
            if frame_range[1] > n_frames:
                sys.exit('time_range is too long, there is an input file '
                         'lasting for {} seconds'.format(n_frames/frame_rate))
    print('The shortest recording last for {} seconds'.format(cur_short_recording))

    """
        read image data
    """
    for group_idx, this_group_root_dir_list in enumerate(root_dir_group_list):
        for file_idx in [file_idx]:   #range(len(this_group_root_dir_list) - 1):
            root_dir = os.path.join(this_group_root_dir_list[0], this_group_root_dir_list[file_idx + 1])
            fn = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
            rgb_files, rgb_files_root = isxrgb.find_rgb_channels(fn)

            rgb_files_with_path = [os.path.join(root_dir, file) for file in rgb_files]

            # open one channel first to get the frame numbers
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
                frame_rate = tmp.frame_rate
                tmp.close()

            # get an example frame to get accurate frame_shape
            # (especially necessary when correct_bad_pixels == True
            tmp = isxrgb.get_rgb_frame(rgb_files_with_path, 0, correct_stray_light=correct_stray_light,
                                       correct_bad_pixels=correct_bad_pixels)
            frame_shape = tmp.shape[1:3]
            n_pixels = frame_shape[0] * frame_shape[1]

            frame_range = np.array(time_range) * frame_rate
            step = math.ceil(frame_range[1]/max_n_frame)
            select_frame_idx = np.arange(frame_range[0], frame_range[1], step)
            rgb_frame_stack = np.zeros((len(ch), frame_shape[0], frame_shape[1], len(select_frame_idx)))
            print('Collect frame', end='')
            for i, frameIdx in enumerate(select_frame_idx):
                print('...', end='')
                this_rgb_frame = isxrgb.get_rgb_frame(rgb_files_with_path, frameIdx,
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

    """
        calculate parameters
    """
    # get tissue name
    tissue = [rgb_frame_file_group_info[i]['tissue'][file_idx] for i in range(n_group)]
    print('Tissue: {}'.format(tissue))
    # find the group of the files for selected led
    led = ['Blue', 'Lime']
    n_led = len(led)
    led_group_idx = mu.find([rgb_frame_file_group_info[i]['led_name'][file_idx] for i in range(n_group)], led)
    tmp = []
    for i in range(n_group):
        tmp += led_group_idx[i]
    led_group_idx = tmp
    pled = [rgb_frame_file_group_info[i]['led_power'][file_idx] for i in led_group_idx]

    # calculate a1r, a1g, a1b, a2r, a2g, a2b (see document)
    # red channel (index=0) and Blue LED (index=0) will be the numerator
    F_de = rgb_frame_file_group[0, :, :, file_idx, led_group_idx[0]]
    a = np.empty((n_ch, n_led))
    aff = np.empty((n_ch, frame_shape[0], frame_shape[1], n_led))      # full frame
    for i in range(n_led):
        for j in range(n_ch):
            F_nu = rgb_frame_file_group[j, :, :, file_idx, led_group_idx[i]]
            tmp = (F_nu/F_de) * (pled[0]/pled[i])
            aff[j, :, :, i] = tmp
            a[j, i] = np.mean(tmp)

    cssp = {'tissue': tissue,
            'led': led,
            'channel': ch,
            'dimension_name': ['channel', 'led'],
            'a': a,
            # 'aff': aff,
            'info': {'root_dir_group_list': root_dir_group_list,
                     'correct_stray_light': correct_stray_light,
                     'time_range': time_range,
                     'time_range_unit': 'second'}}

    """
        save result
    """
    save_filename = 'cssp_{}.json'.format(tissue[0])
    save_filename_with_path = ('{}/result/json/{}'.format(os.getcwd(), save_filename))
    with open(save_filename_with_path, 'w') as f:
        json.dump(cssp, f, cls=mu.NumpyEncoder)
    f.close()

    """
        load the json file and check the result
    """
    filename = save_filename    #'cssp_GcaMP_300s.json'  #'cssp_RGeco_300s.json'    #
    save_filename_with_path = '{}/result/json/{}'.format(os.getcwd(), filename)
    view_calc_params_for_rgb_signal_split_result(save_filename_with_path)

    plt.show()


def view_calc_params_for_rgb_signal_split_result(cssp_filename_with_path):
    """
        View the parameters

        :param cssp_filename_with_path: json file
        :return:
    """

    with open(cssp_filename_with_path) as json_data:
        d = json.load(json_data)

        d['aff'] = np.array(d['aff'])
        ch = d['channel']
        led = d['led']
        shape = d['aff'].shape
        n_ch = shape[0]
        n_row = shape[1]
        n_col = shape[2]
        n_led = shape[3]

        fig = plt.figure(figsize=(8, 5))  # (7, 10))
        gs = plt.GridSpec(n_ch, n_led, wspace=0.4)
        hax_im = [0 for i in range(n_ch)]
        hax_cb = [0 for i in range(n_ch)]
        hax_stat = [0 for i in range(n_ch)]
       # hax_im = [[0 for i in range(n_ch)] for j in range(n_led)]
       #  hax_cb = [[0 for i in range(n_ch)] for j in range(n_led)]
       #  hax_stat = [[0 for i in range(n_ch)] for j in range(n_led)]

        for led_idx in range(n_led):
            for ch_idx in range(n_ch):
                hax0 = plt.subplot(gs[ch_idx, led_idx])
                hax00 = mp.split_axes(hax0, 3, 'horizontal', ratio=[20, 1, 10]) #, gap=0.1) #[0.1, 0.2, 0])
                hax_im[ch_idx] = hax00[0]
                hax_cb[ch_idx] = hax00[1]
                hax_stat[ch_idx] = hax00[2]

            hax_im[0].set_title('{} LED'.format(led[led_idx]))
            """
                show image for an parameter to inspect spatial pattern
            """
            isxrgb.show_rgb_frame(d['aff'][:, :, :, led_idx], ax_list=hax_im, cmap='jet')
            for ch_idx in range(n_ch):
                """
                    show stat for each image
                """
                plt.sca(hax_stat[ch_idx])
                tmp = d['aff'][ch_idx, :, :, led_idx].flatten()
                # tmp = (tmp - np.mean(tmp))/np.std(tmp)
                if ch_idx or led_idx:
                    plt.hist(tmp, 50,  normed=1, facecolor=ch[ch_idx], orientation='horizontal')
                    plt.autoscale(tight=True)
                else:
                    hax_stat[ch_idx].set_visible(False)
                """
                   show colorbar for each image
                """
                clim = hax_im[ch_idx].get_images()[0].get_clim()
                print('this clim is {}'.format(clim))
                ylim = hax_stat[ch_idx].get_ylim()
                print('this ylim is {}'.format(ylim))
                ticks = hax_stat[ch_idx].get_yticks()

                cbar = plt.colorbar(hax_im[ch_idx].get_images()[0], cax=hax_cb[ch_idx], ticks=ticks)
                if led_idx == 0 and ch_idx == 0:
                    ticklabels = [1]
                else:
                    ticklabels = []
                    hax_im[ch_idx].set_xticks([])
                    hax_im[ch_idx].set_yticks([])
                cbar.ax.yaxis.set_ticklabels(ticklabels)

                hax_im[ch_idx].set_ylabel('Param', color=ch[ch_idx])
                # hax_cb[ch_idx].set_ylabel('param', color=ch[ch_idx])
                # hax_cb[ch_idx].yaxis.set_label_position('left')

        filename = os.path.basename(cssp_filename_with_path)
        basename = os.path.splitext(filename)[0]
        figure_title = '{} pixel-based parameters'.format(basename)
        plt.suptitle(figure_title)
        plt.savefig('{}/result/figure/{}'.format(os.getcwd(), basename), dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None)


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
        gs = plt.GridSpec(n_ch)
        # ax_list = []
        ax_list.append(plt.subplot(gs[i]) for i in range(n_ch))

    # plot scatter plot predicted vs measured
    for ch_idx in range(n_ch):
        ax = ax_list[ch_idx]
        plt.sca(ax)
        plt.scatter(rgb_frame_file_measured.reshape([n_ch, n_pixels, -1])[ch_idx, select_pixels, :],
                    rgb_frame_file_predicted.reshape([n_ch, n_pixels, -1])[ch_idx, select_pixels, :], s=2)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        plt.plot(ax.get_xlim(), ax.get_ylim(), 'k-', lw=0.5)
        ax.set_ylabel('Predicted by addition', color=ch[ch_idx])
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
