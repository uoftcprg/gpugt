from sys import argv, stdin

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ITERATION_TIME_PLOT_FILENAME = argv[1]
MEMORY_USAGE_PLOT_FILENAME = argv[2]
CUDA_MEMORY_USAGE_PLOT_FILENAME = argv[3]


def main():
    df = pd.read_csv(stdin, index_col=0)

    plt.clf()
    sns.lineplot(
        df,
        x='Game size (# nodes)',
        y='Iteration time (s)',
        hue='Solver',
    )
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Iteration time versus game size')
    plt.savefig(ITERATION_TIME_PLOT_FILENAME)

    plt.clf()
    sns.lineplot(
        df,
        x='Game size (# nodes)',
        y='Memory usage (bytes)',
        hue='Solver',
    )
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Memory usage versus game size')
    plt.savefig(MEMORY_USAGE_PLOT_FILENAME)

    plt.clf()
    sns.lineplot(
        df[df['CUDA memory usage (bytes)'] != 0],
        x='Game size (# nodes)',
        y='CUDA memory usage (bytes)',
    )
    plt.xscale('log')
    plt.yscale('log')
    plt.title('CUDA memory usage versus game size')
    plt.savefig(CUDA_MEMORY_USAGE_PLOT_FILENAME)


if __name__ == '__main__':
    main()
