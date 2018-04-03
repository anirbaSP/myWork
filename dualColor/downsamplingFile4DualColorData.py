import isx
import os
from os import listdir
from os.path import isfile, join

import numpy as np
# always initialize the API before use
isx.initialize()

root_dir = '/Volumes/Data2/Alice/NV3_DualColor/NV3_color_sensor_12bit/V3-55/V3_55_20170717'
# '/Volumes/data2/Alice/NV3_DualColor/NV3_color_sensor_12bit/V3_39/20170807'
# '/ariel/data2/Alice/NV3_DualColor/NV3_color_sensor_12bit/V3-17/20170717'
# '/Users/Sabrina/workspace/data/pipeline_s'

fn = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
print('{} files have been found, they are \n {}'.format(len(fn), fn))

channel = ['red', 'green', 'blue']

recording_file = fn[2]
recording_file = os.path.join(root_dir, recording_file)
# 'Movie_2017-07-17-16-20-59_blue.isx')
# 'Movie_2017-08-07-14-01-28_blue')
# 'Movie_2017-07-17-10-39-25_blue.isxd') # do same for blue and red chanel

# for later use, grab the name of the recording file, and strip off the file extension
root_dir, recording_file_name_with_ext = os.path.split(recording_file)
recording_name, recording_ext = os.path.splitext(recording_file_name_with_ext)

# print(recording_file, '\n', root_dir, '\n',  recording_file_name_with_ext, '\n', recording_name, '\n', recording_ext)

raw_movie = isx.Movie(recording_file)

print('Frame rate: {:0.2f} Hz'.format(raw_movie.frame_rate))
print('#of frames: {}'.format(raw_movie.num_frames))
print('Movie size: {} rows by {} colums'.format(raw_movie.shape[0], raw_movie.shape[1]))

## plot to see if reading frame is normal"
# frame = raw_movie.read_frame(0)
# plt.figure()
# plt.imshow(frame, aspect='auto')
# plt.show()


## preprocess
pipeline_dir = os.path.join(root_dir, 'pipeline_s') # generate pipeline directory
if not os.path.exists(pipeline_dir):
    os.mkdir(pipeline_dir)

pp_file = os.path.join(pipeline_dir, '{}-PP.isxd'.format(recording_name)) # preprocessed file
print(pp_file)

crop_rect = [10, 10, raw_movie.shape[0] - 10, raw_movie.shape[1] - 1] # crop by 10 pixels
raw_movie.close()
isx.preprocess(recording_file, pp_file, temporal_downsample_factor=5, spatial_downsample_factor=6,
               crop_rect=crop_rect, fix_defective_pixels=True)

