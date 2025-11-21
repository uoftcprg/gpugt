from json import load
from statistics import fmean
from sys import argv, stdout

from scipy.stats import sem
from tqdm import tqdm
import pandas as pd

SOLVER_NAMES = {
    'gpugt.regret_minimizers.CounterfactualRegretMinimization': (
        'Sequence-form (GPU)'
    ),
    'noregret.regret_minimizers.CounterfactualRegretMinimization': (
        'Sequence-form (CPU)'
    ),
    'pyspiel.CFRSolver': 'Classical (C++)',
}
ITERATION_COUNT = int(argv[1])
DATA_PATHNAMES = argv[2:]


def main():
    state_counts = []
    solver_names = []
    times = []
    time_stderrs = []
    used_bytes = []
    ru_maxrsss = []
    lookup = {}

    for pathname in tqdm(DATA_PATHNAMES):
        with open(pathname) as file:
            data = load(file)

        if 'game_path' in data:
            if data['game_path'] in lookup:
                state_count = lookup[data['game_path']]
            else:
                with open(data['game_path']) as file:
                    game = load(file)

                lookup[data['game_path']] = game['state_count']
                lookup[game['game_name']] = game['state_count']
                state_count = game['state_count']
        else:
            state_count = lookup[data['game_name']]

        if 'regret_minimizer_import_string' in data:
            solver_name = SOLVER_NAMES[data['regret_minimizer_import_string']]
        else:
            solver_name = SOLVER_NAMES[data['solver_import_string']]

        times_ = data['times'][:ITERATION_COUNT]

        state_counts.append(state_count)
        solver_names.append(solver_name)
        times.append(fmean(times_))
        time_stderrs.append(sem(times_).item())
        used_bytes.append(data.get('used_bytes', 0))
        ru_maxrsss.append(data['ru_maxrss'])

    data = {
        'Game size (# nodes)': state_counts,
        'Solver': solver_names,
        'Iteration time (s)': times,
        'Iteration time standard error (s)': time_stderrs,
        'CUDA memory usage (bytes)': used_bytes,
        'Memory usage (bytes)': ru_maxrsss,
    }
    df = pd.DataFrame(data)

    df.to_csv(stdout)


if __name__ == '__main__':
    main()
