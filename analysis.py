import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_data(log_file_name):
    data = pd.read_csv(log_file_name)
    ax = plt.gca()
    data.plot(kind='line', y='avg', ax=ax)
    data.plot(kind='line', y='max', color='red', ax=ax)
    data.plot(kind='line', y='min', color='green', ax=ax)

    plt.show()


# get all files in the directory output/ and plot them
def plot_all_files():
    files = os.listdir("./output/")
    for file in files:
        plot_data("./output/" + file)


plot_all_files()
