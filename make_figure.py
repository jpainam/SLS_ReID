import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#sns.set(font_scale=1.2)
#import seaborn as sns
#sns.set()
from scipy.io import savemat, loadmat
import numpy
from glob import glob

k_colors = ["g", "b", "y", "r", "c", "g", "#FFA500", "k", "#77FF55", "m"];
k_markers = "v*do";
fig = plt.figure(figsize=(50, 5.195))


def cmc_figure(mat, keys=None, rank_lim=None, ax=None, first=True, x_label='Rank'):
    methods = {"Dense": mat['dense'],
               "Resnet": mat['resnet'],
               "SLS_Dense": mat['sls_dense'],
               "SLS_Resnet": mat['sls_resnet']}
    legfont = mpl.font_manager.FontProperties(family="monospace")
    # mlabdefaults()
    r1 = []
    result = []
    if keys is None:
        keys = methods.keys()
    for i, m in enumerate(keys):
        y = methods[m]
        x = np.arange(y.size)
        r1.append(y[0])
        ax.plot(x, y,
                color=k_colors[i],
                linestyle="-",
                marker=k_markers[i],
                markevery=5)
        result.append(y)
    ax.set_xlabel(x_label)
    if first:
        ax.set_ylabel("Identification Rate(%)")
    if rank_lim is None:
        ax.set_xlim(0, 60)
    else:
        ax.set_xlim(0, rank_lim)
    ax.set_ylim(75, 100)
    # ax.set_yticks(np.arange(0.75, 1.1, 0.25))
    ax.set_yticks([75, 80, 85, 90, 95, 100])
    r1 = ['%0.2f' % r for r in r1]
    digitlen = len("10.10")
    total_len = len("Euclidean 10.10")

    r1 = ['%s%s(%s%s%%)' % (leg,
                            ''.join([' '] * (total_len - digitlen - len(leg))),
                            ''.join([' '] * (digitlen - len(r))),
                            r) for r, leg in zip(r1, keys)]
    ax.legend(r1, prop=legfont, loc=4)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')


if __name__ == "__main__":
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['savefig.dpi'] = 150
    mpl.rcParams['font.size'] = 10.
    #mpl.rcParams['font.family'] = "Times New Roman"
    mpl.rcParams['legend.fontsize'] = "small"
    mpl.rcParams['legend.fancybox'] = True
    #mpl.rcParams['lines.markersize'] = 10
    #mpl.rcParams['figure.figsize'] = 8, 100.6
    mpl.rcParams['legend.labelspacing'] = 0.01
    mpl.rcParams['legend.borderpad'] = 0.1
    mpl.rcParams['legend.borderaxespad'] = 0.2
    mpl.rcParams['font.monospace'] = "Courier New"

    keys = ['Resnet', "SLS_Resnet", "Dense", 'SLS_Dense']

    mat = loadmat('./model/market.mat')
    ax1 = fig.add_subplot(141)
    cmc_figure(mat, keys=keys, ax=ax1, first=True, x_label='(a)Market')
    ax2 = fig.add_subplot(142)
    mat = loadmat('./duke/duke.mat')
    cmc_figure(mat, keys=keys, ax=ax2, first=False, x_label='(b)Duke')
    ax3 = fig.add_subplot(143)
    mat = loadmat('./cuhk03/cuhk03.mat')
    cmc_figure(mat, keys=keys, ax=ax3, first=False, x_label='(c)CUHK03')
    ax4 = fig.add_subplot(144)
    mat = loadmat('./viper/viper.mat')
    cmc_figure(mat, keys=keys, ax=ax4, first=False, x_label='(d)VIPeR')

    fig.savefig("cmc_curve.eps")
    fig.savefig("cmc_curve.png")
    plt.show()
    plt.clf()
