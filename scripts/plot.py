from sys import argv, stdin

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PLOT_FILENAME = argv[1]


def main():
    df = pd.read_csv(stdin, index_col=0)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    sns.lineplot(
        df,
        x='Game size (# nodes)',
        y='Iteration time (s)',
        hue='Solver',
        style='Implementation',
        ax=axes[0],
        legend=False,
    )
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_title('Iteration time versus game size')
    sns.lineplot(
        df,
        x='Game size (# nodes)',
        y='Memory usage (bytes)',
        hue='Solver',
        style='Implementation',
        ax=axes[1],
        legend=False,
    )
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_title('Memory usage versus game size')
    sns.lineplot(
        df,
        x='Game size (# nodes)',
        y='CUDA memory usage (bytes)',
        hue='Solver',
        style='Implementation',
        ax=axes[2],
    )
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].set_title('CUDA memory usage versus game size')
    sns.move_legend(axes[2], 'center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(PLOT_FILENAME)


if __name__ == '__main__':
    main()
