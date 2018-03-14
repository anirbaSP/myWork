import os
from os import listdir
from os.path import isfile, join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import isx
import isxrgb


def main():

    run_signal_split()
    # run_test_data()
    # run_exp_data()
    # spatial_bandpass_movie()


def run_signal_split():
    root_dir = '/Volumes/data2/Alice/NV3_DualColor/D_Lab/' \
               'Masa/20170816/led12'    #NV3_color_sensor_12bit/V3_71/20171127/led1'
    select_frames = [100]

    fn = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
    print('{} files have been found, they are \n {}'.format(len(fn), fn))

    # find the rgb channel, and sort fn as r/g/b
    rgb_files, rgb_files_root = isxrgb.find_rgb_channels(fn)
    print('rgb files are \n {}'.format(rgb_files))

    rgb_files_with_path = [os.path.join(root_dir, file) for file in rgb_files]

    for i, frame_idx in enumerate(select_frames):
        this_frame = isxrgb.get_rgb_frame(rgb_files_with_path, frame_idx, correct_bad_pixels=True)
        if i == 0:
            shape = this_frame.shape
            stack = np.empty([shape[0], shape[1], shape[2], len(select_frames)])
        stack[:, :, :, i] = this_frame

    plt.figure()
    plt.imshow(stack[1, :, :, 0])

    figure_name = 'x_all'
    plt.savefig(figure_name)

    rgb_files_root = isxrgb.find_rgb_channels(rgb_files)[1]
    rgb_files_root = os.path.split(rgb_files_root)[1]
    exp = isxrgb.get_exp_label(rgb_files_root)
    rgb_s, xyz = isxrgb.rgb_signal_split(stack[:, :, :, 0], exp)

    # show the result
    tissue = list(rgb_s.keys())
    shape = rgb_s[tissue[0]].shape
    n_ch = shape[0]
    n_row = shape[1]
    n_col = shape[2]
    n_led = shape[3]
    for i, this_tissue in enumerate(tissue):
        for j in range(n_led):
            isxrgb.show_rgb_frame(rgb_s[this_tissue][:, :, :, j], clim=[0, 5000])
            plt.suptitle('{}, led{}'.format(this_tissue, j+1))

    isxrgb.show_rgb_frame(xyz)

    plt.show()

    # compare the results with separate LED illumination data sets


def run_test_data():
    root_dir = '/Volumes/data2/Sabrina/NV3_01_greenpixel_bitsdrop/'
    filename = 'v3-01_gr_0ff.isxd' #'v3-01_gb_0ff.isxd' #'v3-01_blue_0ff.isxd' #'v3-01_red_0ff.isxd' #'Movie_2018-01-30-12-53-38_red_only.isxd'  #'Movie_2018-01-30-12-56-08_blue_only.isxd'  #'img_2018-03-08-15-41-21 A80043050047 colorbars.tif' #'img_2018-03-08-15-43-46 A80043050039 colorbars.tif'
    # #'A80043050039.tif' #'Movie_2018-01-30-12-55-17_greenr_raw.isxd' #'Movie_2018-01-30-12-56-52_greenb_raw.isxd' #'both_green_fff_2016-02-11-14-15-10_raw.isxd' #
    select_frames = [0]
    microscope_name = 'NV3-1' #'JH_cropEdge'

    filename_with_path = os.path.join(root_dir, filename)
    ext = os.path.splitext(filename_with_path)[1]

    if ext == '.tif':
        this_movie = Image.open(filename_with_path)
        n_row = this_movie.size[1]
        n_col = this_movie.size[0]
        n_frame = this_movie.n_frames
    elif ext == '.isxd':
        this_movie = isx.Movie(filename_with_path)
        # print('Frame rate:{:0.2f} Hz'.format(this_movie.frame_rate))
        n_row = this_movie.shape[0]
        n_col = this_movie.shape[1]
        n_frame = this_movie.num_frames
    print('#of frames:{}'.format(n_frame))
    print('Movie size: {} rows by {} columns'.format(n_row, n_col))

    stack = np.empty((n_row, n_col, len(select_frames)))
    # get first frame from each channel
    for i, frame_idx in enumerate(select_frames):
        if ext == '.tif':
            this_movie.seek(frame_idx)
            stack[:, :, i] = np.array(this_movie)
            this_movie.close()
        elif ext == '.isxd':
            stack[:, :, i] = this_movie.read_frame(frame_idx)
    this_movie.close()

    # n_colorPan_y_rng = {'w': [3, 155], 'y': [156, 315], 'c': [316, 475], 'g': [476, 635],
    #                     'm': [636, 795], 'r': [796, 955], 'b': [956, 1115], 'k': [1116, 1275]}
    # select_colorPan = 'g'
    # u, u_inverse, u_count = np.unique(stack[:, np.arange(n_colorPan_y_rng[select_colorPan][0],
    #                 n_colorPan_y_rng[select_colorPan][1]+1)], return_inverse=True, return_counts=True)

    # stack = stack[2:-2, 1:-1, :]
    u, u_inverse, u_count = np.unique(stack[:, :, 0], return_inverse=True, return_counts=True)

    # add pixel value and the corresponding count
    label = []
    for i in range(len(u)):
        label.append('{} (n={})'.format(int(u[i]), u_count[i]))
        if u[i] == 0:
            label[i] = '{}'.format(int(u[i]))

    fig = plt.figure(figsize=(8, 8))
    gs = plt.GridSpec(1, 2, width_ratios=[20, 1], right=0.8, top=0.5, bottom=0.05)
    hax = []
    for i in range(2):
        hax.append(plt.subplot(gs[i]))

    plt.suptitle('{} \n{}'.format(microscope_name, filename))
    plt.sca(hax[0])
    im = plt.imshow(stack[:20, :30, 0], cmap='jet', aspect='equal')  #0:20, 0:30
    # hax[0].xaxis.set_ticks(range(0, 30, 5))
    # hax[0].yaxis.set_ticks(range(0, 20, 5))

    # plt.sca(hax[1])
    # hax[1].set_ylim(u[[0, -1]])
    # s=''
    # for i in range(len(u)-1, -1, -1):
    #     s = '{}\n{}'.format(s, label[i])
    # plt.text(2, 0, '{}'.format(s), fontsize=8)

    cbar = plt.colorbar(im, cax=hax[1])  # , location='right', orientation='vertical')
    pos = hax[0].get_position()
    pos0 = hax[1].get_position()
    hax[1].set_position([pos0.x0, pos.y0, pos0.width, pos.height])

    cbar.set_ticks(u[1::])
    cbar.set_ticklabels(label[1::])

    figure_name = '{}/figures/{}_{}'.format(os.getcwd(), microscope_name, filename[0:-5])
    plt.savefig(figure_name, dpi=600, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0,
                frameon=None)

    plt.show()


def run_exp_data():
    root_dir = '/Volumes/data2/Alice/NV3_DualColor/NV3_color_sensor_12bit/' \
                'V3_71/20171127/led1'
    select_frames = [10]

    fn = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
    print('{} files have been found, they are \n {}'.format(len(fn), fn))

    # find the rgb channel, and sort fn as r/g/b
    rgb_files, rgb_files_root = isxrgb.find_rgb_channels(fn)
    print('rgb files are \n {}'.format(rgb_files))

    rgb_files_with_path = [os.path.join(root_dir, file) for file in rgb_files]

    for i, frame_idx in enumerate(select_frames):
        this_frame = isxrgb.get_rgb_frame(rgb_files_with_path, frame_idx, correct_bad_green_pixels=False)
        if i == 0:
            shape = this_frame.shape
            stack = np.empty([shape[0], shape[1], shape[2], len(select_frames)])
        stack[:, :, :, i] = this_frame

    plt.figure()
    plt.imshow(stack[1, :, :, 0])

    figure_name = 'x_all'
    plt.savefig(figure_name)


def spatial_bandpass_movie():
    root_dir = '/Volumes/data2/Alice/NV3_DualColor/K_Lab/Cohort1/SO171214A'

    fn = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
    print('{} files have been found, they are \n {}'.format(len(fn), fn))

    # find the rgb channel, and sort fn as r/g/b
    rgb_files, rgb_files_root = isxrgb.find_rgb_channels(fn)
    print('rgb files are \n {}'.format(rgb_files))

    rgb_files_with_path = [os.path.join(root_dir, file) for file in rgb_files]

    for i, thisfile in enumerate(rgb_files_with_path):
        isx.spatial_bandpass(thisfile, join(root_dir, 'spatial_bandpass_movie', rgb_files[i]), low_bandpass_cutoff=0.005,
                             high_bandpass_cutoff=0.500, retain_mean=False, subtract_global_minimum=True)


if __name__ == '__main__':
    main()



















