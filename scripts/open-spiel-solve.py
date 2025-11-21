from argparse import ArgumentParser
from json import dump
from pathlib import Path
from resource import getrusage, RUSAGE_SELF
from sys import stdout
from time import time

from noregret.utilities import import_string
from pyspiel import exploitability, load_game
from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('game_name')
    parser.add_argument('solver_import_string')
    parser.add_argument('iteration_count', type=int)
    parser.add_argument('-e', '--exploitabilities', type=Path)

    return parser.parse_args()


def main():
    args = parse_args()
    game = load_game(args.game_name)
    solver_type = import_string(args.solver_import_string)
    solver = solver_type(game)
    checkpoint = 1
    iterations = []
    times = []
    exploitabilities = []

    for iteration in trange(1, args.iteration_count + 1):
        begin_time = time()

        solver.evaluate_and_update_policy()

        end_time = time()
        time_ = end_time - begin_time

        if iteration == checkpoint and args.exploitabilities:
            average_policy = solver.average_policy()
            exploitability_ = exploitability(game, average_policy)
            checkpoint *= 2
        else:
            exploitability_ = None

        iterations.append(iteration)
        times.append(time_)
        exploitabilities.append(exploitability_)

    ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    data = {'Iteration': iterations, 'Exploitability': exploitabilities}
    df = pd.DataFrame(data)

    if args.exploitabilities:
        plt.clf()
        sns.lineplot(df, x='Iteration', y='Exploitability')
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'Exploitability of {args.game_name} in self-play')
        plt.savefig(args.exploitabilities)

    data = {
        'game_name': args.game_name,
        'solver_import_string': args.solver_import_string,
        'iteration_count': args.iteration_count,
        'iterations': iterations,
        'times': times,
        'exploitabilities': exploitabilities,
        'ru_maxrss': ru_maxrss,
    }

    dump(data, stdout)


if __name__ == "__main__":
    main()
