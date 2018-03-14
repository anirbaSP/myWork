import isx
import isxrgb
import mymodels as mm

import os
from os import listdir
from os.path import isfile, join

import numpy as np
from PIL import Image
# import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp

import random
import pickle


def main():
    correct_bad_green_pixels = False
    time_range = [0, 1]  # second

    root_dir_group_list = [['/Volumes/data2/Alice/NV3_DualColor/NV3_color_sensor_12bit/'
                            'Scope_Autofluorescence/NV3-04', 'led1_2', 'led1_1', 'led1_0'],
                           ['/Volumes/data2/Alice/NV3_DualColor/NV3_color_sensor_12bit/'
                            'Scope_Autofluorescence/NV3-04', 'led2_5', 'led2_4', 'led2_3', 'led2_2_new',
                            'led2_2', 'led2_1', 'led2_0']
                           ]
    microscope_name = 'NV3-04'

    save_filename_straylight_data = '{}_straylight_data'.format(microscope_name)
    save_filename_straylight = '{}_straylight'.format(microscope_name)

    # create_stray_light_fitting_files(root_dir_group_list, time_range,
    #                                  save_filename_straylight_data, save_filename_straylight,
    #                                  correct_bad_green_pixels=correct_bad_green_pixels)

    view_stray_light_standard_curve(save_filename_straylight_data, save_filename_straylight)


def create_stray_light_fitting_files(root_dir_group_list, time_range,
                                     save_filename_straylight_data, save_filename_straylight,
                                     channel=None, correct_bad_green_pixels=None,
                                     ):

    if channel is None:
        ch = ['red', 'green', 'blue']
    if correct_bad_green_pixels is None:
        correct_bad_green_pixels = False

    straylight = {}
    straylight['file_info'] = []
    max_n_file = max(len(this_group_root_dir_list) for this_group_root_dir_list in root_dir_group_list) - 1

    for group_idx, this_group_root_dir_list in enumerate(root_dir_group_list):
        straylight['file_info'].append({})
        straylight['file_info'][group_idx]['tissue'] = []
        straylight['file_info'][group_idx]['microscope'] = []
        straylight['file_info'][group_idx]['led_name'] = []
        straylight['file_info'][group_idx]['led_power'] = []
        straylight['file_info'][group_idx]['filename'] = []

        for file_idx in range(len(this_group_root_dir_list) - 1):
            root_dir = os.path.join(this_group_root_dir_list[0], this_group_root_dir_list[file_idx + 1])
            fn = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
            print('{} files have been found, they are \n {}'.format(len(fn), fn))

            # find the rgb channel, and sort fn as r/g/b
            rgb_files, rgb_files_root = isxrgb.find_rgb_channels(fn)
            print('rgb files are \n {}'.format(rgb_files))

            exp_label = isxrgb.get_exp_label(rgb_files_root)

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
            # (especially necessary when correct_bad_green_pixels == True
            tmp = isxrgb.get_rgb_frame(rgb_files_with_path, 0, correct_bad_green_pixels=correct_bad_green_pixels)
            frame_shape = tmp.shape[1:3]
            n_pixels = frame_shape[0] * frame_shape[1]

            frame_range = np.array(time_range) * frame_rate
            select_frame_idx = np.arange(frame_range[0], frame_range[1])
            rgb_frame_stack = np.zeros((len(ch), frame_shape[0], frame_shape[1], len(select_frame_idx)))
            print('Collect frame', end='')
            for i, frameIdx in enumerate(select_frame_idx):  # show randomly selected frames
                print('...', end='')
                this_rgb_frame = isxrgb.get_rgb_frame(rgb_files_with_path, frameIdx,
                                                      correct_bad_green_pixels=correct_bad_green_pixels)
                rgb_frame_stack[:, :, :, i] = this_rgb_frame
                print('frame {}'.format(frameIdx))

            # calculate average intensity across time for R/G/B
            rgb_frame_stack_mean4time = np.mean(rgb_frame_stack, axis=3)
            # rgb_frame_stack_std4time = np.std(rgb_frame_stack, axis=3)

            if file_idx == 0 and group_idx == 0:
                rgb_frame_file_group = np.empty((len(ch), frame_shape[0], frame_shape[1],
                                                 max_n_file, len(root_dir_group_list)))
                rgb_frame_file_group.fill(np.nan)

            rgb_frame_file_group[:, :, :, file_idx, group_idx] = rgb_frame_stack_mean4time

            # get experimental parameter from exp_label
            idx1 = exp_label.rfind('(')
            idx2 = exp_label.rfind(')')
            straylight['file_info'][group_idx]['led_power'].append(
                float(exp_label[idx1 + 1:idx2 - 2].replace(',', '.', 1)))
            idx3 = exp_label[0:idx1].rfind(',')
            idx4 = exp_label[0:idx3].rfind(',')
            straylight['file_info'][group_idx]['tissue'].append(exp_label[0:idx4])
            straylight['file_info'][group_idx]['microscope'].append(exp_label[idx4 + 2:idx3])
            straylight['file_info'][group_idx]['led_name'].append(exp_label[idx3 + 2:idx1 - 1])
            straylight['file_info'][group_idx]['filename'].append(rgb_files_root)

        led_power = straylight['file_info'][group_idx]['led_power']
        led_name = straylight['file_info'][group_idx]['led_name'][0]
        straylight[led_name] = {}

        # fit F - Power(LED) for each pixel
        x = led_power
        b1 = np.empty(n_pixels)
        b0 = np.empty(n_pixels)
        straylight[led_name]['b0'] = np.empty((len(ch), frame_shape[0], frame_shape[1]))
        straylight[led_name]['b1'] = np.empty((len(ch), frame_shape[0], frame_shape[1]))
        for i in range(len(ch)):
            tmp = np.reshape(rgb_frame_file_group[i, :, :, :, group_idx], [n_pixels, -1])
            for count, j in enumerate(range(n_pixels)):
                if np.mod(count, 10000) == 0:
                    print('fitting pixel # {} ...'.format(j))
                y = tmp[j, :]
                y = y[~np.isnan(y)]
                coeff, y_hat = mm.linear_regression(x, y)  # fit with linear
                b0[j] = coeff[0]
                b1[j] = coeff[1]
                # y_hat = m * x + b
            straylight[led_name]['b0'][i, :, :] = b0.reshape([frame_shape[0], frame_shape[1]])
            straylight[led_name]['b1'][i, :, :] = b1.reshape([frame_shape[0], frame_shape[1]])

    # save the result into files
    save_data_file = open(save_filename_straylight_data, 'wb')
    pickle.dump(rgb_frame_file_group, save_data_file)
    save_data_file.close()

    save_file = open(save_filename_straylight, 'wb')
    pickle.dump(straylight, save_file)
    save_file.close()


def view_stray_light_standard_curve(filename_straylight_data, filename_straylight, channel=None):

    if channel is None:
        ch = ['red', 'green', 'blue']

    save_data_file = open(filename_straylight_data, 'rb')
    rgb_frame_file_group = pickle.load(save_data_file)
    save_data_file.close()

    save_file = open(filename_straylight, 'rb')
    straylight = pickle.load(save_file)
    save_file.close()

    # get basic info
    tmp = rgb_frame_file_group.shape
    n_ch = tmp[0]
    frame_shape = tmp[1:3]
    n_pixels = frame_shape[0] * frame_shape[1]
    n_group = tmp[4]

    # prepare figure and axes
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(3, n_group)
    ax_list = []
    for i in range(n_ch):
        for j in range(n_group):
            ax_list.append(plt.subplot(gs[i, j]))
    tmp = [ax_list[i:i + n_group] for i in range(0, len(ax_list) - 1, n_group)]
    ax_list = tmp

    # select pixels to show
    n_select_pixels = 5
    random_pixels = False
    if random_pixels:
        select = random.sample(range(n_pixels), n_select_pixels)
    else:
        select = np.arange(0, n_pixels - 1, int(n_pixels / n_select_pixels))

    # plot by group, by channel
    cmap = plt.get_cmap('jet')
    for group_idx in range(n_group):
        led_name = straylight['file_info'][group_idx]['led_name'][0]
        b0 = straylight[led_name]['b0']
        b1 = straylight[led_name]['b1']
        for ch_idx in range(n_ch):
            ax = ax_list[ch_idx][group_idx]
            plt.sca(ax)
            this_b0 = b0[ch_idx, :, :].reshape(n_pixels)
            this_b1 = b1[ch_idx, :, :].reshape(n_pixels)
            for i, pixel_idx in enumerate(select):
                color = cmap(float(i)/n_select_pixels)
                x = straylight['file_info'][group_idx]['led_power']
                y = rgb_frame_file_group.reshape([n_ch, n_pixels, -1, n_group])[ch_idx, pixel_idx, :, group_idx]
                y = y[~np.isnan(y)]
                plt.plot(x, y, '.', c=color)
                y_hat = this_b0[pixel_idx] + this_b1[pixel_idx] * np.array(x)
                plt.plot(x, y_hat, '-', c=color, label='$y = %0.2f x + %0.2f$' % (this_b1[pixel_idx], this_b0[pixel_idx]))
            ax.legend(bbox_to_anchor=(0.005, 0.995), loc=2, borderaxespad=0)

            if group_idx == 0:
                plt.ylabel(ch[ch_idx], color=ch[ch_idx])
            ax.set_xticks(x)  #straylight['file_info'][group_idx]['led_power'])
            # if group_idx in set(n_group + np.array([-1, 0])):
            #     ax.set_xticklabels('')

        if ch_idx == 2:
            if group_idx < 2:
                xlabel_string = '{} LED'.format(straylight['file_info'][group_idx]['led_name'][0])
            # elif group_idx == 2:
            #     xlabel_string = 'Blue LED + Lime LED'
            # else:
            #     xlabel_string = 'Predicted by addition'
            plt.xlabel(xlabel_string)

    #
    # # plot scatter plot predicted vs measured
    # select10000 = np.arange(0, n_pixels - 1, 10000)
    # for ch_idx in range(len(ch)):
    #     ax = ax_list[ch_idx][-1]
    #     plt.sca(ax)
    #     plt.scatter(rgb_frame_file_group.reshape(shape0)[select10000, ch_idx, :, -2],
    #                 rgb_frame_file_group.reshape(shape0)[select10000, ch_idx, :, -1], s=2)
    #     ax.set_aspect('equal', 'box')
    #     ax.autoscale(tight=True)
    #     plt.plot(ax.get_xlim(), ax.get_ylim(), 'k-', lw=0.5)
    #     ax.set_ylabel('Predicted', color=ch[ch_idx])
    #     if ch_idx == 2:
    #         ax.set_xlabel('Measured')

    # plt.show()
    figure_title = '{}_{}_standard_curve_byPixel'.format(straylight['file_info'][group_idx]['tissue'][0],
                                                         straylight['file_info'][group_idx]['microscope'][
                                                             0])

    htitle = plt.suptitle('{}, {} example pixels'.format(figure_title, n_select_pixels))
    gs.tight_layout(fig, rect=[0.02, 0.02, 0.96, 0.95])
    plt.savefig('{}/figures/{}'.format(os.getcwd(), figure_title), dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None)
    plt.show()


if __name__ == '__main__': main()