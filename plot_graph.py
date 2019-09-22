import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_saved_file(filename):
    data = pd.read_csv(filename + ".tsv", sep='\t', header=None)
    data.rename(columns={0:'iter_index', 1:'diff'}, inplace=True)
    data['iter_index'] = data['iter_index'] / 100000
    plot = data.plot(x='iter_index', y='diff', linewidth=0.1, legend=None)
    plot.set_xlabel("Simulation Iteration 10^5")
    plot.set_ylabel("Q-value difference")
    plt.ylim([0, 0.5])
    plt.title(filename[17:-8])
    plt.xticks(np.arange(0, 11, 1.0))
    plt.savefig(filename + ".png")


plot_saved_file('SoccerGamePlayersFriendQ_1000000')