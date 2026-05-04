from sys import argv

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PLOT_FILENAME = argv[1]
TOTAL_TIME = int(argv[2])


def main():
    plt.rcParams['legend.fontsize'] = 15
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['font.size'] = 15

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for i, (game, path) in enumerate(zip(argv[3::2], argv[4::2])):
        df = pd.read_csv(path, index_col=0)
        sns.lineplot(
            df,
            x='Wall-clock time (s)',
            y='Exploitability',
            hue='Solver',
            style='Implementation',
            ax=axes[i],
            legend=False,
        )
        # axes[i].set_xscale('log')
        axes[i].set_xlim((-TOTAL_TIME * 0.05, TOTAL_TIME * 1.05))
        axes[i].set_yscale('log')
        axes[i].set_title(game)

    fig.tight_layout()
    fig.savefig(PLOT_FILENAME)


if __name__ == '__main__':
    main()
