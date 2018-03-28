# !/usr/local/bin/python
# isxrgb.py
import isx


def get_rgb_frame(rgb_files, frame_idx, camera_bias=None, correct_stray_light=None, correct_bad_pixels=None):
    import os
    from PIL import Image
    import numpy as np

    # get movie info for each channel
    if camera_bias is None:
        camera_bias = 172
    if correct_stray_light is None:
        correct_stray_light = False

    if correct_bad_pixels is None:
        correct_bad_pixels = False

    ext = os.path.splitext(rgb_files[0])[1]
    if ext == '.isxd':  # for .isxd files
        # # For debug purpose: check if all channels are correct
        # for this_file in rgb_files:
        #     this_movie = isx.Movie(this_file)
        #     print('Frame rate:{:0.2f} Hz'.format(this_movie.frame_rate))
        #     print('#of frames:{}'.format(this_movie.num_frames))
        #     print('Movie size: {} rows by {} columns'.format(this_movie.shape[0], this_movie.shape[1]))
        #     this_movie.close()
        #     # may need to handle cases when the files detected are not red/green/blue channels, ect.

        this_movie = isx.Movie(rgb_files[0])
        n_row = this_movie.shape[0]
        n_col = this_movie.shape[1]
        this_movie.close()

        n_ch = len(rgb_files)
        rgb_frame = np.empty([n_ch, n_row, n_col])

        for i, this_file in enumerate(rgb_files):
            this_movie = isx.Movie(this_file)
            frame_idx = int(frame_idx)
            this_frame = this_movie.read_frame(frame_idx)
            rgb_frame[i, :, :] = this_frame - camera_bias
            this_movie.close()

    elif ext == '.tif':
        # # For debug purpose: check if all channels are correct
        # for this_file in rgb_files:
        #     this_movie = Image.open(this_file)
        #     print('Frame rate:{:0.2f} Hz'.format(this_movie.frame_rate))
        #     print('#of frames:{}'.format(this_movie.n_frames))
        #     print('Movie size: {} rows by {} columns'.format(this_movie.size[1], this_movie.size[0]))

        this_movie = Image.open(rgb_files[0])
        n_row = this_movie.size[1]
        n_col = this_movie.size[0]
        this_movie.close()

        n_ch = len(rgb_files)
        rgb_frame = np.zeros((n_ch, n_row, n_col))

        for i, this_file in enumerate(rgb_files):
            this_movie = Image.open(this_file)
            this_movie.seek(frame_idx)
            rgb_frame[i, :, :] = np.array(this_movie)
            this_movie.close()

    rgb_files_root = find_rgb_channels(rgb_files)[1]
    rgb_files_root = os.path.split(rgb_files_root)[1]
    exp = get_exp_label(rgb_files_root)
    exp_label = exp['label']

    if correct_stray_light:
        rgb_frame = subtract_stray_light(rgb_frame, exp_label)

    if correct_bad_pixels:
        rgb_frame = discard_bad_pixels(rgb_frame)

    return rgb_frame


def discard_bad_pixels(rgb):
    """
        NV3-01 microscope has pixels with dropping bits therefor the reading is not correct. This function can discard
        bad pixels. In order to recognize bad pixels, test patterns have been acquired ahead of time and the files were
        saved. Any information about the test pattern files are stored in LOOKUP_bad_pixels.json
    :param rgb: [n_channel, n_row, m_col] array
    :param exp_label: a string with information about the microscope
    :return: rgb_out: array after correct bad pixels, with the shape [n_channel, nn_row, mm_col]
    """

    import numpy as np
    import json

    shape = rgb.shape
    n_ch = shape[0]
    n_row = shape[1]
    n_col = shape[2]

    downsampling_x = 2
    downsampling_y = 4
    downsampling = max(downsampling_x, downsampling_y)
    rgb_out = np.empty([n_ch, int(n_row / downsampling), int(n_col / downsampling)])
    # red channel
    x_keep = np.arange(0, n_row, downsampling_x)
    y_keep = np.arange(1, n_col, downsampling_y)
    tmp = rgb[0, x_keep, :]
    tmp = tmp[:, y_keep]
    tmp[:, 0] = np.nan
    rgb_out[0, :, :] = np.nanmean(np.stack((tmp[::2, :], tmp[1::2, :]), axis=2), axis=2)

    # blue channel
    x_keep = np.arange(1, n_row, downsampling_x)
    y_keep = np.arange(0, n_col, downsampling_y)
    tmp = rgb[2, x_keep, :]
    tmp = tmp[:, y_keep]
    rgb_out[2, :, :] = np.mean(np.stack((tmp[::2, :], tmp[1::2, :]), axis=2), axis=2)

    # green channel
    # greenr
    x_keep = np.arange(0, n_row, downsampling_x)
    y_keep = np.arange(0, n_col, downsampling_y)
    tmp = rgb[1, x_keep, :]
    tmp = tmp[:, y_keep]
    tmp1 = np.mean(np.stack((tmp[::2, :], tmp[1::2, :]), axis=2), axis=2)
    # greenb
    x_keep = np.arange(1, n_row, downsampling_x)
    y_keep = np.arange(1, n_col, downsampling_y)
    tmp = rgb[1, x_keep, :]
    tmp = tmp[:, y_keep]
    tmp2 = np.mean(np.stack((tmp[::2, :], tmp[1::2, :]), axis=2), axis=2)

    rgb_out[1, :, :] = np.mean(np.stack((tmp1, tmp2), axis=2), axis=2)

    # first column is nan for red channel, discard it
    rgb_out = rgb_out[:, :, 1::]

    # data = json.load(open('LOOKUP_bad_pixels.json'))  # todo: add create json file script
    # idx = exp_label.find(',')
    # key = exp_label[idx + 2:]
    # bad_pixels_file_base = data[key]
    #
    # ch = ['red', 'green', 'blue']
    # bad_pixels_files = list()
    # for i, channel in enumerate(ch):
    #     bad_pixels_files.append('{}_{}.isxd'.format(bad_pixels_file_base, channel))
    #
    # bad_pixels = get_bad_pixels_frame(bad_pixels_files)
    #
    # if rgb.ndim == 4:
    #     rgb_out = np.subtract(np.moveaxis(rgb, 3, 0), bad_pixels)
    #     rgb_out = np.moveaxis(rgb, 0, -1)
    # else:
    #     rgb_out = np.subtract(rgb, bad_pixels)

    return rgb_out


class discard_bad_pixels(object):
    """
        a dispatch method to handle bad pixels for data collected with NV3-01 color sensor
    """

    import numpy as np

    def discard_bad_pixels_for_channel(self, channel, im):
        # Dispatch method
        method_name = 'discard_bad_pixels_for_' + str(channel) + 'channel'
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: "Invalid channel")
        # Call the method as we return it

        return method()


    def discard_bad_pixels_for_red_channel(self, channel, im):
        x_keep = np.arange(0, n_row, downsampling_x)
        y_keep = np.arange(1, n_col, downsampling_y)
        tmp = rgb[0, x_keep, :]
        tmp = tmp[:, y_keep]
        tmp[:, 0] = np.nan
        rgb_out[0, :, :] = np.nanmean(np.stack((tmp[::2, :], tmp[1::2, :]), axis=2), axis=2)


    def discard_bad_pixels_for_green_channel(self, channel, im):
        # greenr
        x_keep = np.arange(0, n_row, downsampling_x)
        y_keep = np.arange(0, n_col, downsampling_y)
        tmp = rgb[1, x_keep, :]
        tmp = tmp[:, y_keep]
        tmp1 = np.mean(np.stack((tmp[::2, :], tmp[1::2, :]), axis=2), axis=2)
        # greenb
        x_keep = np.arange(1, n_row, downsampling_x)
        y_keep = np.arange(1, n_col, downsampling_y)
        tmp = rgb[1, x_keep, :]
        tmp = tmp[:, y_keep]
        tmp2 = np.mean(np.stack((tmp[::2, :], tmp[1::2, :]), axis=2), axis=2)


    def discard_bad_pixels_for_blue_channel(self, channel, im):
        x_keep = np.arange(1, n_row, downsampling_x)
        y_keep = np.arange(0, n_col, downsampling_y)
        tmp = rgb[2, x_keep, :]
        tmp = tmp[:, y_keep]
        rgb_out[2, :, :] = np.mean(np.stack((tmp[::2, :], tmp[1::2, :]), axis=2), axis=2)






def discard_bad_pixels(frame, channel):
    """
       current only for NV3-01 color sensor
    :param frame:
    :param channel:
    :return:
    """

    import numpy as np


    shape = rgb.shape
    n_ch = shape[0]
    n_row = shape[1]
    n_col = shape[2]

    downsampling_x = 2
    downsampling_y = 4
    downsampling = max(downsampling_x, downsampling_y)
    rgb_out = np.empty([n_ch, int(n_row / downsampling), int(n_col / downsampling)])
    # red channel
    x_keep = np.arange(0, n_row, downsampling_x)
    y_keep = np.arange(1, n_col, downsampling_y)
    tmp = rgb[0, x_keep, :]
    tmp = tmp[:, y_keep]
    tmp[:, 0] = np.nan
    rgb_out[0, :, :] = np.nanmean(np.stack((tmp[::2, :], tmp[1::2, :]), axis=2), axis=2)

    # blue channel
    x_keep = np.arange(1, n_row, downsampling_x)
    y_keep = np.arange(0, n_col, downsampling_y)
    tmp = rgb[2, x_keep, :]
    tmp = tmp[:, y_keep]
    rgb_out[2, :, :] = np.mean(np.stack((tmp[::2, :], tmp[1::2, :]), axis=2), axis=2)

    # green channel
    # greenr
    x_keep = np.arange(0, n_row, downsampling_x)
    y_keep = np.arange(0, n_col, downsampling_y)
    tmp = rgb[1, x_keep, :]
    tmp = tmp[:, y_keep]
    tmp1 = np.mean(np.stack((tmp[::2, :], tmp[1::2, :]), axis=2), axis=2)
    # greenb
    x_keep = np.arange(1, n_row, downsampling_x)
    y_keep = np.arange(1, n_col, downsampling_y)
    tmp = rgb[1, x_keep, :]
    tmp = tmp[:, y_keep]
    tmp2 = np.mean(np.stack((tmp[::2, :], tmp[1::2, :]), axis=2), axis=2)

    rgb_out[1, :, :] = np.mean(np.stack((tmp1, tmp2), axis=2), axis=2)

    # first column is nan for red channel, discard it
    rgb_out = rgb_out[:, :, 1::]

    # data = json.load(open('LOOKUP_bad_pixels.json'))  # todo: add create json file script
    # idx = exp_label.find(',')
    # key = exp_label[idx + 2:]
    # bad_pixels_file_base = data[key]
    #
    # ch = ['red', 'green', 'blue']
    # bad_pixels_files = list()
    # for i, channel in enumerate(ch):
    #     bad_pixels_files.append('{}_{}.isxd'.format(bad_pixels_file_base, channel))
    #
    # bad_pixels = get_bad_pixels_frame(bad_pixels_files)
    #
    # if rgb.ndim == 4:
    #     rgb_out = np.subtract(np.moveaxis(rgb, 3, 0), bad_pixels)
    #     rgb_out = np.moveaxis(rgb, 0, -1)
    # else:
    #     rgb_out = np.subtract(rgb, bad_pixels)

    return rgb_out


def write_corrected_isxd_movie(input_filename_with_path, correct_bad_pixel=None, correct_stray_light=None,
                               save_pathname=None, save_filename=None):
    """
        rewrite the movie to an .isxd file after different options, such as discard bad pixels, correct microscope stray
        light, etc.
    :param input_filename_with_path: input .isxd or .tif movie
    :param correct_bad_pixel:
    :param correct_stray_light:
    :param save_pathname:
    :param save_filename:
    :return:
    """

    from os.path import split, splitext, exists, join
    from os import mkdir

    pathname, filename = split(input_filename_with_path)
    basename, ext = splitext(filename)

    if save_pathname is None:
        save_pathname = join(pathname, 'movie_corrected')
    if ~exists(save_pathname):
        mkdir(save_pathname)

    if save_filename is None:
        save_filename = filename

    if correct_bad_pixel is None:
        correct_bad_pixel = False
    if correct_stray_light is None:
        correct_stray_light = False

    mov_in = isx.Movie(input_filename_with_path)
    flag = True
    frameIdx = 0
    while flag:
        im_in = mov_in.read_frame(frameIdx)
        if correct_bad_pixel:









def get_bad_pixels_frame(bad_pixels_files, frame_idx):
    pass    # todo


def find_rgb_channels(fn, channel_list=None):  # find the files for RGB channels

    import os
    import myutilities as mu

    if channel_list is None:
        channel_list = ['red', 'green', 'blue']

    ch_order = mu.find(fn, channel_list)
    rgb_files = [fn[i[0]] for i in ch_order]

    rgb_files_root = os.path.commonprefix(rgb_files)
    rgb_files_root = rgb_files_root[:-1]

    return rgb_files, rgb_files_root


def find_rgb_files(basename_with_path, channel_list=None):

    """
        Use rgb file basename to find files for all channels
    :param basename_with_path:
    :param channel_list: default ['red', 'green', 'blue']
    :return:
    """

    from os.path import split, isfile

    if channel_list is None:
        channel_list = ['red', 'green', 'blue']
    n_ch = len(channel_list)

    tmp = split(basename_with_path)
    pathname = tmp[0]
    basename = tmp[1]

    rgb_files_with_path = [0]*n_ch
    for i in range(n_ch):
        rgb_files_with_path[i] = '{}_{}.isxd'.format(basename_with_path, channel_list[i])    #todo:
        if not isfile(rgb_files_with_path[i]):
            return 'rgb file for {} channel does not exist'.format(channel_list[i])

    return rgb_files_with_path


def get_movie_header(rgb_files_with_path, correct_bad_pixels, correct_stray_light):  #todo: rewrite to change it as a class
    """
        get the movie "header" info
    :param rgb_files_with_path:
    :return: size(n_row, n_col, n_frame), and frame_rate
    """
    from os.path import splitext
    from PIL import Image
    # open one channel first to get the basics of the movie
    ext = splitext(rgb_files_with_path[0])[1]

    if ext == '.isxd':
        tmp = isx.Movie(rgb_files_with_path[0])
        # frame_shape = tmp.shape
        num_frames = tmp.num_frames
        frame_rate = tmp.frame_rate
        frame_period = tmp.get_frame_period()
        data_type = tmp.data_type
        tmp.close()
    elif ext == '.tif':
        tmp = Image.open(rgb_files_with_path[0])
        # frame_shape = tmp.size[::-1]
        num_frames = tmp.n_frames
        frame_rate = 20  # tmp.frame_rate #todo: does the tif file always have frame rate info? what's the tag name??
        frame_period = 50000
        tmp.close()

    # get an example frame to get accurate frame_shape (especially necessary when correct_bad_pixels == True
    tmp = get_rgb_frame(rgb_files_with_path, 0, correct_stray_light=correct_stray_light,
                        correct_bad_pixels=correct_bad_pixels)
    frame_shape = tmp.shape[1:3]

    return frame_shape, num_frames, frame_period, data_type, frame_rate


def get_exp_label(exp_file_root):   #todo: perhaps need to re-structure the create_LOOKUP_exp_label file

    import json

    data = json.load(open('LOOKUP_exp_label.txt'))
    label = data[exp_file_root]

    idx1 = label.rfind('(')
    idx2 = label.rfind(')')
    led_power = float(label[idx1 + 1:idx2 - 2].replace(',', '.', 1))
    idx3 = label[0:idx1].rfind(',')
    idx_tmp1 = label[0:idx3].rfind('(')
    idx_tmp2 = label[0:idx3].rfind(')')
    if idx_tmp1 != -1 or idx_tmp2 != -1:
        idx3 = label[0:idx3].rfind(',')
    idx4 = label[0:idx3].rfind(',')
    tissue = label[0:idx4]
    microscope = label[idx4 + 2:idx3]
    led_name = label[idx3 + 2:idx1 - 1]

    exp = {'label': label,
           'microscope': microscope,
           'tissue': tissue,
           'led_name': led_name,
           'led_power': led_power,
           'Blue': [],
           'Lime': []
           }

    # find Blue and Lime LED seperately
    idx_blue = label.find('Blue')
    if idx_blue != -1:
        idx1 = label.find('(', idx_blue)
        idx2 = label.find(')', idx_blue)
        exp['Blue'] = float(label[idx1 + 1:idx2 - 2].replace(',', '.', 1))
    idx_lime = label.find('Lime')
    if idx_lime != -1:
        idx1 = label.find('(', idx_lime)
        idx2 = label.find(')', idx_lime)
        exp['Lime'] = float(label[idx1 + 1:idx2 - 2].replace(',', '.', 1))

    return exp


def subtract_stray_light(rgb, exp_label, correct_stray_light=None):

    import numpy as np
    import json
    import pickle
    import myutilities as mu

    # # find microscope_name, led_name and led_power
    # idx1 = exp_label.rfind('(')
    # idx2 = exp_label.rfind(')')
    # led_power = float(exp_label[idx1 + 1:idx2 - 2].replace(',', '.', 1))
    # idx3 = exp_label[0:idx1].rfind(',')
    # idx4 = exp_label[0:idx3].rfind(',')
    # tissue = exp_label[0:idx4]
    # microscope_name = exp_label[idx4 + 2:idx3]
    # led_name = exp_label[idx3 + 2:idx1 - 1]
    #
    # # idx1 = exp_label.find(',')
    # # idx2 = exp_label[idx1:].find(',')
    # # microscope_name = exp_label[idx + 2: idx2 - 1]
    #
    # straylight_data_file = open('{}_straylight_data'.format(microscope_name), 'rb')
    # rgb_frame_file_group = pickle.load(straylight_data_file)
    # straylight_data_file.close()
    #
    # straylight_file = open('{}_straylight'.format(microscope_name), 'rb')
    # straylight = pickle.load(straylight_file)
    # straylight_file.close()
    #
    # # find microscope_name, led_name and led_power
    # n_group = len(straylight['file_info'])
    # group_idx = [straylight['file_info'][i]['led_name'][0] for i in range(n_group)].index(led_name)
    # file_idx = straylight['file_info'][group_idx]['led_power'].index(led_power)
    #
    # stray_frame = rgb_frame_file_group[:, :, :, file_idx, group_idx]
    # rgb = np.subtract(rgb, stray_frame)

    if correct_stray_light is None:
        correct_stray_light = False

    data = json.load(open('LOOKUP_stray_light_correction.txt'))
    idx = exp_label.find(',')
    key = exp_label[idx + 2:]
    stray_light_file_base = data[key]

    ch = ['red', 'green', 'blue']
    stray_light_files = list()
    for i, channel in enumerate(ch):
        stray_light_files.append('{}_{}.isxd'.format(stray_light_file_base, channel))

    frame_idx = 10
    stray_frame = get_rgb_frame(stray_light_files, frame_idx, correct_bad_pixels=correct_bad_pixels)

    if rgb.ndim == 4:
        rgb = np.subtract(np.moveaxis(rgb, 3, 0), stray_frame)
        rgb = np.moveaxis(rgb, 0, -1)
    else:
        rgb = np.subtract(rgb, stray_frame)

    return rgb


def get_rgb_pixel_time(rgb_files_with_path, select_frame_idx=None, select_pixel_idx=None, correct_stray_light=None,
                       correct_bad_pixels=None):

    import os
    from PIL import Image
    import numpy as np

    # Get intensity for specific pixels at specific frames
    ext = os.path.splitext(rgb_files_with_path[0])[1]
    if ext == '.isxd':
        tmp = isx.Movie(rgb_files_with_path[0])
        frame_shape = tmp.shape
        n_pixels = frame_shape[0] * frame_shape[1]
        n_frames = tmp.num_frames
        tmp.close()
    elif ext == '.tif':
        tmp = Image.open(rgb_files_with_path[0])
        frame_shape = tmp.size[::-1]
        n_pixels = frame_shape[0] * frame_shape[1]
        n_frames = tmp.n_frames
    else:
        frame_shape = []
        n_frames = []
        n_pixels = []

    if select_frame_idx is None:
        select_frame_idx = np.arange(0, n_frames - 1)
    if select_pixel_idx is None:
        select_pixel_idx = np.arange(0, n_pixels - 1)
    if correct_stray_light is None:
        correct_stray_light = False
    if correct_bad_pixels is None:
        correct_bad_pixels = False

    rgb_pixel_time = np.empty([3, len(select_pixel_idx), len(select_frame_idx)])
    print('Collect frame', end='')
    this_rgb_frame = np.empty([frame_shape[0], frame_shape[1]])
    for i, frame_idx in enumerate(select_frame_idx):
        print('...', end='')
        this_rgb_frame = get_rgb_frame(rgb_files_with_path, frame_idx, correct_stray_light=correct_stray_light,
                                       correct_bad_pixels=correct_bad_pixels)
        rgb_pixel_time[:, :, i] = this_rgb_frame.reshape([-1, n_pixels])[:, select_pixel_idx]
        print('frame {}'.format(frame_idx))

    return rgb_pixel_time


def calc_rgb_ratio(rgb_data, ch=None):

    import numpy as np

    from itertools import combinations

    if ch is None:
        ch = ['red', 'green', 'blue']
    comb = combinations(range(len(ch)), 2)
    pair_list = list(comb)
    tmp = list(rgb_data.shape[1:])
    tmp.insert(0, len(pair_list))
    rgb_ratio = np.empty(tmp)
    for i in range(len(pair_list)):
        rgb_ratio[i, :, :] = rgb_data[pair_list[i][0]] / rgb_data[pair_list[i][1]]

    return rgb_ratio, pair_list


def rgb_signal_split(rgb, aM_inv):
    """
        apply aM_inv to the raw rgb frame to get XYZ for the frame
    :param rgb:
    :param aM_inv:
    :return:
    """
    import numpy as np

    n_ch = rgb.shape[0]
    n_row = rgb.shape[1]
    n_col = rgb.shape[2]
    n_pixel = n_row * n_col
    assert aM_inv.shape[0] == n_ch
    n_probe = aM_inv.shape[1]

    # r_alpha, r_beta, r_gama,
    # g_alpha, g_beta, g_gama,
    # b_alpha, b_beta, b_gama

    # rgb_s = np.empty([n_ch, n_pixel, n_led, n_probe])
    xyz = np.empty([n_probe, n_pixel])
    # for i in range(n_pixel):
    xyz = np.matmul(aM_inv, rgb.reshape([n_ch, n_pixel]))
        # for j, this_probe in enumerate(probe):
        #     rgb_s[:, :, :, i][:, i, :] = np.multiply(q[:, j, :], tmp[j])
        # xyz[:, i] = tmp

    # reshape pixels back to image
    # for j, this_probe in enumerate(probe):
    #     rgb_s[this_probe] = rgb_s[this_probe].reshape([n_ch, n_row, n_col, n_led])
    xyz = xyz.reshape([n_probe, n_row, n_col])

    return xyz


def write_cssp_movie(rgb_filename_with_path, save_pathname=None, save_filename=None,
                     correct_stray_light=None, correct_bad_pixels=None):
    """
        Write GCaMP and RGeco movie after color signal splitting.

    :param rgb_files_with_path:
    :param save_path:
    :param save_filename:
    :return:
    """
    from os.path import dirname, basename, exists, join
    from os import remove
    import numpy as np

    rgb_files_basename = basename(rgb_filename_with_path)

    # todo: rewrite get_exp_label function, make it as a class with method like exp.led, exp.probe
    # exp = get_exp_label(rgb_files_basename)
    # exp.led.blue = 1
    # exp.led.lime = 0.5
    # exp.probe = ['GCaMP', 'RGeco']
    # aM = calc_aMatrix_for_rgb_signal_split(exp.probe, exp.led, cssp=None)
    exp_led = [1.8, 0.9]
    exp_probe = ['GCaMP', 'RGeco', 'Autofluo']
    aM = calc_aMatrix_for_rgb_signal_split(exp_led, cssp=None)
    aM_inv = np.linalg.inv(aM)

    if save_pathname is None:
        save_pathname = dirname(rgb_filename_with_path)
    if save_filename is None:
        save_filename = rgb_files_basename
    save_filename_with_path = [join(save_pathname, '{}_{}.isxd'.format(save_filename, probe)) for probe in exp_probe]
    if correct_stray_light is None:
        correct_stray_light = False
    if correct_bad_pixels is None:
        correct_bad_pixels = False

    rgb_files_with_path = find_rgb_files(rgb_filename_with_path)

    shape, num_frames, frame_period, data_type, frame_rate= get_movie_header(rgb_files_with_path,
                                                                  correct_bad_pixels=correct_bad_pixels,
                                                                  correct_stray_light=correct_stray_light)
    n_row = shape[0]
    n_col = shape[1]
    n_pixel = n_row * n_col
    n_frame = num_frames
    n_ch = len(rgb_files_with_path)
    n_probe = len(exp_probe)

    output_mov_list = [0]*n_probe
    for i in range(n_probe):
        if exists(save_filename_with_path[i]):
            remove(save_filename_with_path[i])
        output_mov_list[i] = isx.Movie(save_filename_with_path[i], frame_period=frame_period, shape=shape, num_frames=num_frames, data_type=data_type)
    print('Writing frame...')
    for frame_idx in range(n_frame):
        rgb_frame = get_rgb_frame(rgb_files_with_path, frame_idx, correct_stray_light=correct_stray_light,
                                  correct_bad_pixels=correct_bad_pixels)
        xyz = rgb_signal_split(rgb_frame, aM_inv)
        xyz[xyz < 0] = 0
        for i in range(n_probe):
            output_mov_list[i].write_frame(xyz[i, :, :], frame_idx)
        if (frame_idx + 1) / 10 != 0 and (frame_idx + 1) % 10 == 0:
            print('...')
            print('\n'.join(map(str, range(frame_idx - 9, frame_idx + 1))))

    print('... all frames done!')

    for i in range(n_probe):
        output_mov_list[i].close()


def calc_aMatrix_for_rgb_signal_split(pled, cssp=None):
    """
    load parameters for color signal spliting files cssp_GCaMP.json, cssp_RGeco.json, and cssp_Autofluo.json, and
        calculate the matraix for rgb signal spliting with the parameters a and led power
    :param pled: a two element list of led power, [p_blueLED, p_LimeLED]
    :param cssp: a dictionaly
    :return: aM is a n_ch-by-n_probe matrix for color signal splitting
    """
    import json
    import numpy as np
    if cssp is None:
        cssp = {}
        cssp['GCaMP'] = '/ariel/data2/Sabrina/data/result/json/cssp_GCaMP.json'
        cssp['RGeco'] = '/ariel/data2/Sabrina/data/result/json/cssp_RGeco.json'
        cssp['Autofluo'] = '/ariel/data2/Sabrina/data/result/json/cssp_Autofluo.json'

    probe = list(cssp.keys())
    assert probe == ['GCaMP', 'RGeco', 'Autofluo']

    n_ch = 3
    n_probe = 3
    aM = np.empty([n_ch, n_probe])
    for i, thisprobe in enumerate(probe):
        with open(cssp[thisprobe]) as json_data:
            d = json.load(json_data)
        led = d['led']
        ch = d['channel']
        dimension_name = d['dimension_name']
        # todo: error if it's not as asserted
        assert led == ['Blue', 'Lime']
        assert ch == ['red', 'green', 'blue']
        assert dimension_name == ['channel', 'led']
        a = np.array(d['a'])
        aM[:, i] = np.matmul(a, np.transpose(pled))

    return aM


def show_rgb_frame(rgb_frame, ax_list=None, clim=None, cmap=None, colorbar=None, share_colorbar=None):

    import matplotlib.pyplot as plt
    import numpy as np

    # show an rgb frame, 3 panels
    ch = ['red', 'green', 'blue']
    n_ch = len(ch)

    if cmap is None:
        cmapp = ['Reds', 'Greens', 'Blues']
    else:
        cmapp = [cmap, cmap, cmap]

    if colorbar is None:
        colorbar = False

    if share_colorbar is None:
        share_colorbar = False

    if clim == 'all' or (ax_list is None and clim is None):
        clim = (np.nanmin(rgb_frame), np.nanmax(rgb_frame))
        print('This auto all clim is {}'.format(clim))

    if clim == 'all' or ax_list is None:
        colorbar = True
        share_colorbar = True

    if clim is None or \
            (isinstance(clim, list) and len(clim) == 2 and not any([isinstance(clim0, list) for clim0 in clim])):
        clim = [clim] * 3

    if ax_list is None:
        fig = plt.figure(figsize=(10, 3))
        gs = plt.GridSpec(1, n_ch+1, width_ratios=[12, 12, 12, 1], wspace=0.1)
        ax_list = list()
        for k in range(n_ch + 1):
            ax_list.append(plt.subplot(gs[0, k]))

    for i, hax in enumerate(ax_list[0:n_ch]):
        plt.sca(hax)
        im = plt.imshow(rgb_frame[i, :, :], clim=clim[i], cmap=cmapp[i])  # cmap=plt.get_cmap('gray'),
        # ax.set_aspect('equal')
        if cmap is None and colorbar and not share_colorbar:
            cb = plt.colorbar(im, ticks=clim)  #, location='right', orientation='vertical')
            pos = hax.get_position()
            pos0 = cb.ax.get_position()
            cb.ax.set_position([pos0.x0, pos.y0, pos0.width, pos.height])
            cb.ax.set_adjustable('box-forced')

        # plt.title('{}'.format(ch[i]))
        # plt.autoscale(tight=True)

    if colorbar and share_colorbar and len(ax_list) > n_ch:
        plt.colorbar(cax=ax_list[i+1])

        # cbar.ax.set_yticks(clim)
        # cbar.ax.set_yticklabels(['low', 'medium', 'high'])
