import isx

from os import listdir
from os.path import isfile, join, splitext

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def main():

    root_dir = '/ariel/data2/Alice/NV3_DualColor/D_Lab/Brian/Cohort_1'
    save_path = join(root_dir, 'meanProj_subBg')

    # generate_meanProj_subBg_files(root_dir, save_path, 'red')
    # generate_meanProj_subBg_files(root_dir, save_path, 'green')
    # generate_meanProj_subBg_files(root_dir, save_path, 'green', dff=True)

    mouse_subdir = ['mouse15', 'mouse20', 'mouse30', 'mouse31']
    match_red_to_green(save_path, mouse_subdir)

    plt.show()

def generate_meanProj_subBg_files(root_dir, save_path, channel, dff=None):

    """
        get mean intensity projection image for all red and green images
    """

    if dff is None:
        dff = False

    fn = [f for f in listdir(root_dir) if (isfile(join(root_dir, f)) and '_{}.isxd'.format(channel) in f)]
    print('{} files have been found, they are \n {}'.format(len(fn), fn))

    isx.initialize()
    for i, thisfile in enumerate(fn):
        mov = isx.Movie(join(root_dir, thisfile))
        n_frame = mov.num_frames
        n_frame = min(n_frame, 1000)
        shape = mov.shape
        n_row = shape[0]
        n_col = shape[1]
        tmp = np.empty([n_row, n_col, n_frame])
        for j in range(n_frame):
            tmp[:, :, j] = mov.read_frame(j)

        mov_mean = np.mean(tmp, axis=2)

        if dff:
            mov_dff = np.divide(tmp, mov_mean[:, :, np.newaxis])
            mov_dff_max = np.max(mov_dff, axis=2)
            im = Image.fromarray(mov_dff_max)
            thisfile_basename = '{}_dff'.format(splitext(thisfile)[0])
        else:
            mov_mean_median = np.median(mov_mean.flatten())
            mov_mean = mov_mean - mov_mean_median
            im = Image.fromarray(mov_mean)
            thisfile_basename = splitext(thisfile)[0]

        im.save(join(save_path, thisfile_basename + '.tif'))
        print('the {}th file {} is completed'.format(i, thisfile))
        mov.close()
        im.close()
    isx.shutdown()


def match_red_to_green(pathname, mouse_subdir):

    """
        For selected mouse, match red and green images to find the best match depth
    """
    for i, thismouse_subdir in enumerate(mouse_subdir):
        thismouse_path = join(pathname, thismouse_subdir)
        fn_green = [f for f in listdir(thismouse_path) if (isfile(join(thismouse_path, f)) and '_green_dff.tif' in f)]
        fn_red = [f for f in listdir(thismouse_path) if (isfile(join(thismouse_path, f)) and '_red.tif' in f)]
        print('{} Green file is {}'.format(len(fn_green), fn_green))
        print('{} Red files are {}'.format(len(fn_red), fn_red))

        fn_green_with_path = [join(thismouse_path, file) for file in fn_green]
        fn_red_with_path = [join(thismouse_path, file) for file in fn_red]
        """
        prepare figure to show the red and green image
        """
        # prepare figure
        fig = plt.figure(figsize=(8, 5))  # (7, 10))
        p_row = 3
        p_col = 4
        gs = plt.GridSpec(p_row, p_col)
        count = 0
        hax = [([0] * p_col) for i in range(p_row)]

        # get and show green image
        tmp = Image.open(fn_green_with_path[0])
        tmp.seek(0)
        im0 = np.array(tmp)
        im0.clip(min=0)
        im_g = np.array(im0)
        tmp.close()

        hax[0][0] = plt.subplot(gs[0, 0])
        count += 1
        plt.sca(hax[0][0])
        plt.imshow(im_g, cmap='jet')
        plt.colorbar()

        n_row = im_g.shape[0]
        n_col = im_g.shape[1]

        # get and show red image
        n_red = len(fn_red_with_path)
        im_r_stack = np.empty([n_row, n_col, n_red])
        for j, thisfile_red in enumerate(fn_red_with_path):
            tmp = Image.open(thisfile_red)
            tmp.seek(0)
            im0 = np.array(tmp)
            im0.clip(min=0)
            im_r_stack[:, :, j] = im0
            tmp.close()

            row = count // p_col
            col = count % p_col
            hax[row][col] = plt.subplot(gs[row, col])
            plt.colorbar()
            count += 1
            plt.sca(hax[row][col])
            plt.imshow(im_r_stack[:, :, j], cmap='jet')


if __name__ == '__main__': main()






