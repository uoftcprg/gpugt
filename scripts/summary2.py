from json import load
from sys import argv, stdout

from tqdm import tqdm
import pandas as pd

IMPLEMENTATION_NAMES = {
    'gpugt.regret_minimizers.CounterfactualRegretMinimization': (
        'Sequence-form'
    ),
    'noregret.regret_minimizers.CounterfactualRegretMinimization': (
        'Sequence-form'
    ),
    'pyspiel.CFRSolver': 'Classical',
    'open_spiel.python.algorithms.cfr.CFRSolver': 'Classical',
}
SOLVER_NAMES = {
    'gpugt.regret_minimizers.CounterfactualRegretMinimization': (
        'Ours (parallelized)'
    ),
    'noregret.regret_minimizers.CounterfactualRegretMinimization': (
        'Ours (unparallelized)'
    ),
    'pyspiel.CFRSolver': 'OpenSpiel (C++)',
    'open_spiel.python.algorithms.cfr.CFRSolver': 'OpenSpiel (Python)',
}
DATA_PATHNAMES = argv[1:]


def main():
    implementation_names = []
    solver_names = []
    wc_times = []
    exploitabilities = []

    for pathname in tqdm(DATA_PATHNAMES):
        with open(pathname) as file:
            data = load(file)

        if 'regret_minimizer_import_string' in data:
            implementation_name = (
                IMPLEMENTATION_NAMES[data['regret_minimizer_import_string']]
            )
        else:
            implementation_name = (
                IMPLEMENTATION_NAMES[data['solver_import_string']]
            )

        if 'regret_minimizer_import_string' in data:
            solver_name = SOLVER_NAMES[data['regret_minimizer_import_string']]
        else:
            solver_name = SOLVER_NAMES[data['solver_import_string']]

        for wc_time, exploitability in zip(
                data['wc_times'],
                data['exploitabilities'],
        ):
            if exploitability is not None:
                implementation_names.append(implementation_name)
                solver_names.append(solver_name)
                wc_times.append(wc_time)
                exploitabilities.append(exploitability)

    data = {
        'Implementation': implementation_names,
        'Solver': solver_names,
        'Wall-clock time (s)': wc_times,
        'Exploitability': exploitabilities,
    }
    df = pd.DataFrame(data)

    df.to_csv(stdout)


if __name__ == '__main__':
    main()
