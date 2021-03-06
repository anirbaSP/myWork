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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import json


def main():

    isx.initialize()

    correct_stray_light = False
    correct_bad_pixels = True
    time_range = [0, 300]  # second
    max_n_frame = 20000

    # root_dir_group_list = [['/ariel/data2/Alice/NV3_DualColor/NV3_color_sensor_12bit/V3-17', '20170717', 'tmp'],
    #                        ['/ariel/data2/Alice/NV3_DualColor/NV3_color_sensor_12bit/V3-17', '20170714', 'tmp']
    #                        ]
    # root_dir_group_list = [['/ariel/data2/Alice/NV3_DualColor/NV3_color_sensor_12bit/V3-55', 'V3_55_20170717'],
    #                        ['/ariel/data2/Alice/NV3_DualColor/NV3_color_sensor_12bit/V3-55', '20170710']
    #                        ]
    root_dir_group_list = [['/ariel/data2/Alice/NV3_DualColor/NV3_color_sensor_12bit/V3-63/20170807', 'led1'],
                           ['/ariel/data2/Alice/NV3_DualColor/NV3_color_sensor_12bit/V3-63/20170807', 'led2']
                           ]

    save_pathname = '/ariel/data2/Sabrina/data/result/json/tmp'
    save_filename_with_path = calc_params_for_rgb_signal_split(root_dir_group_list, time_range, max_n_frame,
                                                               correct_stray_light=correct_stray_light,
                                                               correct_bad_pixels=correct_bad_pixels,
                                                               save_pathname=save_pathname,
                                                               save_filename=None)

    """
        load the json file and check the result
    """
    # save_filename = 'cssp_GCaMP.json'
    # save_filename_with_path = '/ariel/data2/Sabrina/data/result/json/update{}'.format(filename)
    view_calc_params_for_rgb_signal_split_result(save_filename_with_path)

    plt.show()

    isx.shutdown()


def calc_params_for_rgb_signal_split(root_dir_group_list, time_range, max_n_frame,
                                     correct_stray_light=None, correct_bad_pixels=None,
                                     save_pathname=None, save_filename=None):

    if correct_bad_pixels is None:
        correct_bad_pixels = False
    if correct_stray_light is None:
        correct_stray_light = False

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

            frame_shape, n_frames, frame_period, data_type, frame_rate = isxrgb.get_movie_header(rgb_files_with_path,
                                         correct_bad_pixels=correct_bad_pixels, correct_stray_light=correct_stray_light)

            frame_range = np.array(time_range) * frame_rate
            cur_short_recording = min(cur_short_recording, n_frames / frame_rate)
            if frame_range[1] > n_frames:
                sys.exit('time_range is too long, there is an input file '
                         'lasting for {} seconds'.format(n_frames / frame_rate))
    print('The shortest recording last for {} seconds'.format(cur_short_recording))

    """
        read image data
    """
    for group_idx, this_group_root_dir_list in enumerate(root_dir_group_list):
        for file_idx in [file_idx]:  # range(len(this_group_root_dir_list) - 1):
            root_dir = os.path.join(this_group_root_dir_list[0], this_group_root_dir_list[file_idx + 1])
            fn = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
            rgb_files, rgb_files_root = isxrgb.find_rgb_channels(fn)

            rgb_files_with_path = [os.path.join(root_dir, file) for file in rgb_files]

            frame_shape, num_frames, frame_period, data_type, frame_rate = isxrgb.get_movie_header(rgb_files_with_path,
                                         correct_bad_pixels=correct_bad_pixels, correct_stray_light=correct_stray_light)

            n_pixels = frame_shape[0] * frame_shape[1]
            frame_range = np.array(time_range) * frame_rate
            step = math.ceil(frame_range[1] / max_n_frame)
            select_frame_idx = np.arange(frame_range[0], frame_range[1], step)
            rgb_frame_stack = np.zeros((len(ch), frame_shape[0], frame_shape[1], len(select_frame_idx)))
            print('Collect frame', end='')
            for i, frameIdx in enumerate(select_frame_idx):
                this_rgb_frame = isxrgb.get_rgb_frame(rgb_files_with_path, frameIdx,
                                        correct_stray_light=correct_stray_light, correct_bad_pixels=correct_bad_pixels)
                rgb_frame_stack[:, :, :, i] = this_rgb_frame
                if (i + 1) / 10 != 0 and (i + 1) % 10 == 0:
                    print('...')
                    print('\n'.join(map(str, select_frame_idx[(i - 9):(i + 1)])))

            # calculate variance across time for R/G/B
            rgb_frame_stack_var4time = np.var(rgb_frame_stack, axis=3)

            # calculate average intensity across time for R/G/B
            rgb_frame_stack_mean4time = np.mean(rgb_frame_stack, axis=3)

            if file_idx == 0 and group_idx == 0:
                shape = rgb_frame_stack.shape
                rgb_frame_file_group = np.empty((shape[0], shape[1], shape[2], max_n_file, n_group))
                rgb_frame_file_group_var = np.empty((shape[0], shape[1], shape[2], max_n_file, n_group))

            rgb_frame_file_group[:, :, :, file_idx, group_idx] = rgb_frame_stack_mean4time
            rgb_frame_file_group_var[:, :, :, file_idx, group_idx] = rgb_frame_stack_var4time


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
    aff = np.empty((n_ch, frame_shape[0], frame_shape[1], n_led))  # full frame
    for i in range(n_led):
        for j in range(n_ch):
            F_nu = rgb_frame_file_group[j, :, :, file_idx, led_group_idx[i]]
            tmp = (F_nu / F_de) * (pled[0] / pled[i])
            aff[j, :, :, i] = tmp
            a[j, i] = np.mean(tmp)

    cssp = {'tissue': tissue,
            'led': led,
            'channel': ch,
            'dimension_name': ['channel', 'led'],
            'a': a,
            'aff': aff,
            'rgb_frame_file_group': rgb_frame_file_group,
            'rgb_frame_file_group_var': rgb_frame_file_group_var,
            'info': {'root_dir_group_list': root_dir_group_list,
                     'correct_stray_light': correct_stray_light,
                     'time_range': time_range,
                     'time_range_unit': 'second'}}

    """
        save result
    """
    if save_pathname is None:
        save_pathname = '/ariel/data2/Sabrina/data/result/json'
    if save_filename is None:
        save_filename = 'cssp_{}.json'.format(tissue[0])

    save_filename_with_path = join(save_pathname, save_filename)
    with open(save_filename_with_path, 'w') as f:
        json.dump(cssp, f, cls=mu.NumpyEncoder)
    f.close()

    return save_filename_with_path


def view_calc_params_for_rgb_signal_split_result(cssp_filename_with_path):
    """
        View the parameters

        :param cssp_filename_with_path: json file
        :return:
    """

    clim_list = [[None, None, None], [None, None, None]]
    clim_list[0][1] = [0, 4000]
    clim_list[0][2] = [0, 4000]
    clim_list[1][0] = [8, 80]
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
        gs = plt.GridSpec(n_ch, n_led, wspace=0.4, left=0.05, right=0.88)
        hax_im = [0 for i in range(n_ch)]
        hax_cb = [0 for i in range(n_ch)]
        hax_stat = [0 for i in range(n_ch)]

        for led_idx in range(n_led):
            for ch_idx in range(n_ch):
                hax0 = plt.subplot(gs[ch_idx, led_idx])
                hax00 = mp.split_axes(hax0, 3, 'horizontal', ratio=[20, 1, 10], gap=[0.1, 0.23, 0])
                hax_im[ch_idx] = hax00[0]
                hax_cb[ch_idx] = hax00[1]
                hax_stat[ch_idx] = hax00[2]

            hax_im[0].set_title('{} LED'.format(led[led_idx]))
            """
                show image for an parameter to inspect spatial pattern
            """
            dd = np.array(d['rgb_frame_file_group_var'])[:, :, :, 0, led_idx]
            # isxrgb.show_rgb_frame(d['aff'][:, :, :, led_idx], ax_list=hax_im, cmap='jet')
            isxrgb.show_rgb_frame(dd, ax_list=hax_im, cmap='jet', clim=clim_list[led_idx])
            for ch_idx in range(n_ch):
                clim = hax_im[ch_idx].get_images()[0].get_clim()
                print('this clim is {}'.format(clim))
                """
                    show stat for each image
                """
                plt.sca(hax_stat[ch_idx])
                tmp = dd[ch_idx, :, :].flatten()
                # tmp = (tmp - np.mean(tmp))/np.std(tmp)

                plt.hist(tmp, 200,  normed=0, facecolor=ch[ch_idx], orientation='horizontal', range=clim)
                plt.autoscale(tight=True)

                # if ch_idx or led_idx:
                #     plt.hist(tmp, 500,  normed=1, facecolor=ch[ch_idx], orientation='horizontal')
                #     plt.autoscale(tight=True)
                # else:
                #     hax_stat[ch_idx].set_visible(False)

                """
                   show colorbar for each image
                """
                clim = hax_im[ch_idx].get_images()[0].get_clim()
                print('this clim is {}'.format(clim))
                ylim = hax_stat[ch_idx].get_ylim()
                print('this ylim is {}'.format(ylim))
                ticks = hax_stat[ch_idx].get_yticks()

                cbar = plt.colorbar(hax_im[ch_idx].get_images()[0], cax=hax_cb[ch_idx], ticks=ticks)

                ticklabels = []
                hax_im[ch_idx].set_xticks([])
                hax_im[ch_idx].set_yticks([])
                cbar.ax.yaxis.set_ticklabels(ticklabels)
                # if led_idx == 0 and ch_idx == 0:
                #     ticklabels = [1]
                # else:
                #     ticklabels = []
                #     hax_im[ch_idx].set_xticks([])
                #     hax_im[ch_idx].set_yticks([])
                # cbar.ax.yaxis.set_ticklabels(ticklabels)

                hax_cb[ch_idx].set_ylabel('var', color=ch[ch_idx])   #'Param'
                hax_cb[ch_idx].yaxis.set_label_coords(-1.5, 0.5)
                # print('yaxis label coords is {}'. format(hax_cb[ch_idx].yaxis.get_label_coords()))
                # hax_cb[ch_idx].yaxis.get_label().set_position((-1000, 0.5))
                # print('yaxis label position is {}'. format(hax_cb[ch_idx].yaxis.get_label().get_position()))
                # hax_cb[ch_idx].set_ylabel('param', color=ch[ch_idx])
                # hax_cb[ch_idx].yaxis.set_label_position('left')

                if led_idx == 0:
                    hax_im[ch_idx].set_ylabel(ch[ch_idx], color=ch[ch_idx])  # 'Param'

        filename = os.path.basename(cssp_filename_with_path)
        basename = os.path.splitext(filename)[0]
        figure_title = '{} pixel-based variance across time, Mice V3-17'.format(basename)   #parameters
        plt.suptitle(figure_title)
        plt.savefig('{}/result/figure/{}'.format(os.getcwd(), basename), dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None)


if __name__ == '__main__': main()
