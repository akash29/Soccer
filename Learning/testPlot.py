import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['agg.path.chunksize'] = 1000
#plt.rcParams["figure.figsize"] = (10, 10)

def moving_average(a, n=2) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot(fileName,figName,title):
    values = np.load(fileName)
    temp = [y - x for x, y in zip(values, values[1:])]
    temp = np.abs(temp)
    temp_avg = moving_average(temp)
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    xmin, xmax = axes.get_xlim()
    axes.set_ylim([ymin, 0.5])
    plt.xlabel("Number of episodes")
    plt.ylabel("Q-value Difference")
    plt.suptitle(title)
    plt.plot(temp_avg)
    plt.savefig(figName)
    plt.show()






