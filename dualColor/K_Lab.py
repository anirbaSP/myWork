import os
from os import listdir
from os.path import isfile, join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import isx
from sklearn.decomposition import FastICA

import isxrgb


def main():

    root_dir = '/Volumes/data2/Alice/NV3_DualColor/K_Lab/Cohort1/SO171214A/spatial_bandpass_movie'
    # root_dir = '/Volumes/data2/Alice/NV3_DualColor/NV3_color_sensor_12bit/Scope_Autofluorescence/NV3-04/led2_4_coverOff'
    correct_stray_light = False
    select_frames = [10]

    fn = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
    print('{} files have been found, they are \n {}'.format(len(fn), fn))

    # find the rgb channel, and sort fn as r/g/b
    rgb_files, rgb_files_root = isxrgb.find_rgb_channels(fn)
    print('rgb files are \n {}'.format(rgb_files))

    rgb_files_with_path = [os.path.join(root_dir, file) for file in rgb_files]

    for i, frame_idx in enumerate(select_frames):
        this_frame = isxrgb.get_rgb_frame(rgb_files_with_path, frame_idx, correct_bad_green_pixels=False,
                                          correct_stray_light=correct_stray_light)
        if i == 0:
            shape = this_frame.shape
            stack = np.empty([shape[0], shape[1], shape[2], len(select_frames)])
        stack[:, :, :, i] = this_frame

    # plt.figure()
    # plt.imshow(stack[0, :, :, 0]) #cmap='gray') #, clim=[350, 500])
    # plt.colorbar()

    isxrgb.show_rgb_frame(stack[:, :, :, 0], cmap='gray', clim=[0, 200])  #'viridis'

    figure_name = 'NV3-04_spatial_bandpass'
    plt.suptitle(figure_name)
    if correct_stray_light:
        figure_name = '{}_correct'.format(figure_name)

    plt.savefig(figure_name)

    # rgb_ica(stack[:, :, :, 0])


def rgb_ica(rgb_frame):
    shape = rgb_frame.shape
    X = rgb_frame.reshape([-1, shape[1]*shape[2]])
    print('X.shape=',X.T.shape)
    ica = FastICA(n_components=2)
    Xtrans = ica.fit_transform(X.T)

    plt.figure()
    for k in range(2):
        ax = plt.subplot(1, 3, k+1)
        img = Xtrans[:, k].reshape([shape[1], shape[2]])
        plt.imshow(img, aspect='auto')
        plt.colorbar()
        plt.title('IC {}'.format(k))
    plt.show()


if __name__ == '__main__':
    main()