import matplotlib.pyplot as plt


def split_axes(ax, n_axes, direction, gap=None, ratio=None):
    """
        Split axes evenly.

        :param ax: an axes object
        :param n_axes: a scalar
        :param direction: 'vertical' or 'horizontal'
        :param gap: a fraction number, or a list such that len(list) = 1 or n_axes
        :param ratio: a list of int specify the ratio among subaxes
        :return: n axes objects
    """
    if gap is None:
        gap = [0.1]
    elif not isinstance(list, gap):
        gap = list(gap)

    if len(gap) == 1:
        gap = gap * n_axes
    elif len(gap) != n_axes:
        print("gap list provided doesn't match with axes number")

    if ratio is None:
        ratio = [1] * n_axes

    pos = ax.get_position().bounds
    ax.remove()
    ax_list = []

    for j in range(n_axes):
        pos0 = []
        if direction == 'vertical':
            pos0.append(pos[0])
            pos0.append(pos[1] + (sum(ratio[:j]) / sum(ratio) + sum(gap[:j])) * pos[3])
            pos0.append(pos[2])
            pos0.append(pos[3] * (ratio[j] / sum(ratio)))

        elif direction == 'horizontal':
            pos0.append(pos[0] + (sum(ratio[:j]) / sum(ratio) + sum(gap[:j])) * pos[2])
            pos0.append(pos[1])
            pos0.append(pos[2] * (ratio[j] / sum(ratio)))
            pos0.append(pos[3])
        ax_list.append(plt.axes(pos0))

    return ax_list
