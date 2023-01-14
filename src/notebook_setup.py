import matplotlib.pyplot as plt
import numpy as np

# https://stackoverflow.com/a/21009774/4591810
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

# Default plotting style
LW = 2
plt.rc('lines', lw=LW)

SMALL_SIZE = 12
MEDIUM_SIZE = 12
BIGGER_SIZE = 14
# https://stackoverflow.com/a/39566040/4591810
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
