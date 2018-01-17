import isx

import numpy as np

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import os

from os import listdir
from os.path import isfile, join

# always initialize the API before use
isx.initialize()

# input files
root_dir = '/ariel/data2/Alice/NV3_DualColor/NV3_color_sensor_12bit/V3-17/20170717/pipeline'
fn = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]

print('{} files have been found, they are \n {}'.format(len(fn), fn))

# find the files for RGB channels
ch = list()
for f in fn:
    # name, ext = os.path.splitext(fn[i])
    idx1 = f.rfind('_')
    idx2 = f.rfind('-')
    color = f[idx1+1:idx2]
    ch.append(color)

print('{} channels have been found, they are {}'.format(len(ch), ch))


# show example frame for each channel
plt.figure()
count = 0
for f in fn:
    this_file = os.path.join(root_dir, f)
    this_movie = isx.Movie(this_file)

    print('Frame rate:{:0.2f} Hz'.format(this_movie.frame_rate))
    print('#of frames:{}'.format(this_movie.num_frames))
    print('Movie size: {} rows by {} colums'.format(this_movie.shape[0], this_movie.shape[1]))

    this_frame0 = this_movie.read_frame(0)
    # fig = plt.figure()

    fig.suptitle('{}'.format(ch[count]))
    plt.imshow(this_frame0, aspect='auto')
    plt.show()

    count += 1
# recording_file = os.path.join(root_dir, fn)  # 'Movie_2017-07-17-10-39-25_red-PP.isxd'

#
# # for later use, grab the name of the recording file, and strip off the file extension
# root_dir, recording_file_name_with_ext = os.path.split(recording_file)
# recording_name, recording_ext = os.path.splitext(recording_file_name_with_ext)
#
# # print(recording_file, '\n', root_dir, '\n',  recording_file_name_with_ext, '\n', recording_name, '\n', recording_ext)
#
# raw_movie = isx.Movie(recording_file)
#
# print('Frame rate: {:0.2f} Hz'.format(raw_movie.frame_rate))
# print('#of frames: {}'.format(raw_movie.num_frames))
# print('Movie size: {} rows by {} colums'.format(raw_movie.shape[0], raw_movie.shape[1]))
#
# ## plot to see if reading frame is normal"
# frame = raw_movie.read_frame(0)
# plt.figure()
# plt.imshow(frame, aspect='auto')
# plt.show()



