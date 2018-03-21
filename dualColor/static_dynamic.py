"""
get mean intensity projection image for all red images
"""
import isx

from os import listdir
from os.path import isfile, join, splitext

import numpy as np
from PIL import Image

root_dir = '/ariel/data2/Alice/NV3_DualColor/D_Lab/Brian/Cohort_1'
save_path = join(root_dir, 'meanProj_subBg')

fn = [f for f in listdir(root_dir) if (isfile(join(root_dir, f)) and '_red.isxd' in f)]
print('{} files have been found, they are \n {}'.format(len(fn), fn))

isx.initialize()
for i, thisfile in enumerate(fn):
    mov = isx.Movie(join(root_dir, thisfile))
    n_frame = mov.num_frames
    shape = mov.shape
    n_row = shape[0]
    n_col = shape[1]
    tmp = np.empty([n_row, n_col, n_frame])
    for j in range(n_frame):
        tmp[:, :, j] = mov.read_frame(j)
    mov_mean = np.mean(tmp, axis=2)

    mov_mean_median = np.median(mov_mean.flatten())
    mov_mean = mov_mean - mov_mean_median

    im = Image.fromarray(mov_mean)
    thisfile_basename= splitext(thisfile)[0]
    im.save(join(save_path, thisfile_basename + '.tif'))



