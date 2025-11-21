# GPUGT

GPU parallelizable implementation of counterfactual regret minimization.

## Scripts

Convert games from OpenSpiel.

```console
python scripts/open-spiel-game.py kuhn_poker | python scripts/compress.py > games/kuhn-poker.json
python scripts/open-spiel-game.py leduc_poker | python scripts/compress.py > games/leduc-poker.json
python scripts/open-spiel-game.py liars_dice | python scripts/compress.py > games/liars-dice.json
python scripts/open-spiel-game.py 'battleship(board_height=2,board_width=2,ship_sizes=[2],ship_values=[1],num_shots=3)' | python scripts/compress.py > games/battleship-tiny.json
python scripts/open-spiel-game.py 'battleship(board_height=3,board_width=2,ship_sizes=[2],ship_values=[1],num_shots=3)' | python scripts/compress.py > games/battleship-small.json
python scripts/open-spiel-game.py 'battleship(board_height=4,board_width=4,ship_sizes=[1],ship_values=[1],num_shots=2)' | python scripts/compress.py > games/battleship-medium.json
python scripts/open-spiel-game.py 'battleship(board_height=3,board_width=3,ship_sizes=[1;2],ship_values=[1;1],num_shots=2)' | python scripts/compress.py > games/battleship-large.json
python scripts/open-spiel-game.py 'battleship(board_height=3,board_width=2,ship_sizes=[2],ship_values=[1],num_shots=4)' | python scripts/compress.py > games/battleship-huge.json
```

Solve games using GPUGT.

```console
python scripts/solve.py \
    games/kuhn-poker.json \
    gpugt.games.TwoPlayerZeroSumExtensiveFormGame \
    gpugt.regret_minimizers.CounterfactualRegretMinimization \
    1024 \
    -a \
    -e figures/gpugt/exploitabilities/kuhn-poker.pdf \
    -v figures/gpugt/values/kuhn-poker.pdf \
    -n "Kuhn poker" \
    > data/gpugt/kuhn-poker.json
python scripts/solve.py \
    games/leduc-poker.json \
    gpugt.games.TwoPlayerZeroSumExtensiveFormGame \
    gpugt.regret_minimizers.CounterfactualRegretMinimization \
    1024 \
    -a \
    -e figures/gpugt/exploitabilities/leduc-poker.pdf \
    -v figures/gpugt/values/leduc-poker.pdf \
    -n "Leduc poker" \
    > data/gpugt/leduc-poker.json
python scripts/solve.py \
    games/liars-dice.json \
    gpugt.games.TwoPlayerZeroSumExtensiveFormGame \
    gpugt.regret_minimizers.CounterfactualRegretMinimization \
    1024 \
    -a \
    -e figures/gpugt/exploitabilities/liars-dice.pdf \
    -v figures/gpugt/values/liars-dice.pdf \
    -n "liar's dice" \
    > data/gpugt/liars-dice.json
python scripts/solve.py \
    games/battleship-tiny.json \
    gpugt.games.TwoPlayerZeroSumExtensiveFormGame \
    gpugt.regret_minimizers.CounterfactualRegretMinimization \
    8 \
    -a \
    -v figures/gpugt/values/battleship-tiny.pdf \
    -n "battleship (tiny)" \
    > data/gpugt/battleship-tiny.json
python scripts/solve.py \
    games/battleship-small.json \
    gpugt.games.TwoPlayerZeroSumExtensiveFormGame \
    gpugt.regret_minimizers.CounterfactualRegretMinimization \
    8 \
    -a \
    -v figures/gpugt/values/battleship-small.pdf \
    -n "battleship (small)" \
    > data/gpugt/battleship-small.json
python scripts/solve.py \
    games/battleship-medium.json \
    gpugt.games.TwoPlayerZeroSumExtensiveFormGame \
    gpugt.regret_minimizers.CounterfactualRegretMinimization \
    8 \
    -a \
    -v figures/gpugt/values/battleship-medium.pdf \
    -n "battleship (medium)" \
    > data/gpugt/battleship-medium.json
python scripts/solve.py \
    games/battleship-large.json \
    gpugt.games.TwoPlayerZeroSumExtensiveFormGame \
    gpugt.regret_minimizers.CounterfactualRegretMinimization \
    8 \
    -a \
    -v figures/gpugt/values/battleship-large.pdf \
    -n "battleship (large)" \
    > data/gpugt/battleship-large.json
python scripts/solve.py \
    games/battleship-huge.json \
    gpugt.games.TwoPlayerZeroSumExtensiveFormGame \
    gpugt.regret_minimizers.CounterfactualRegretMinimization \
    8 \
    -a \
    -v figures/gpugt/values/battleship-huge.pdf \
    -n "battleship (huge)" \
    > data/gpugt/battleship-huge.json
```

Solve games using NoRegret.

```console
python scripts/solve.py \
    games/kuhn-poker.json \
    noregret.games.TwoPlayerZeroSumExtensiveFormGame \
    noregret.regret_minimizers.CounterfactualRegretMinimization \
    1024 \
    -a \
    -e figures/noregret/exploitabilities/kuhn-poker.pdf \
    -v figures/noregret/values/kuhn-poker.pdf \
    -n "Kuhn poker" \
    > data/noregret/kuhn-poker.json
python scripts/solve.py \
    games/leduc-poker.json \
    noregret.games.TwoPlayerZeroSumExtensiveFormGame \
    noregret.regret_minimizers.CounterfactualRegretMinimization \
    1024 \
    -a \
    -e figures/noregret/exploitabilities/leduc-poker.pdf \
    -v figures/noregret/values/leduc-poker.pdf \
    -n "Leduc poker" \
    > data/noregret/leduc-poker.json
python scripts/solve.py \
    games/liars-dice.json \
    noregret.games.TwoPlayerZeroSumExtensiveFormGame \
    noregret.regret_minimizers.CounterfactualRegretMinimization \
    1024 \
    -a \
    -e figures/noregret/exploitabilities/liars-dice.pdf \
    -v figures/noregret/values/liars-dice.pdf \
    -n "liar's dice" \
    > data/noregret/liars-dice.json
python scripts/solve.py \
    games/battleship-tiny.json \
    noregret.games.TwoPlayerZeroSumExtensiveFormGame \
    noregret.regret_minimizers.CounterfactualRegretMinimization \
    8 \
    -a \
    -v figures/noregret/values/battleship-tiny.pdf \
    -n "battleship (tiny)" \
    > data/noregret/battleship-tiny.json
python scripts/solve.py \
    games/battleship-small.json \
    noregret.games.TwoPlayerZeroSumExtensiveFormGame \
    noregret.regret_minimizers.CounterfactualRegretMinimization \
    8 \
    -a \
    -v figures/noregret/values/battleship-small.pdf \
    -n "battleship (small)" \
    > data/noregret/battleship-small.json
python scripts/solve.py \
    games/battleship-medium.json \
    noregret.games.TwoPlayerZeroSumExtensiveFormGame \
    noregret.regret_minimizers.CounterfactualRegretMinimization \
    8 \
    -a \
    -v figures/noregret/values/battleship-medium.pdf \
    -n "battleship (medium)" \
    > data/noregret/battleship-medium.json
python scripts/solve.py \
    games/battleship-large.json \
    noregret.games.TwoPlayerZeroSumExtensiveFormGame \
    noregret.regret_minimizers.CounterfactualRegretMinimization \
    8 \
    -a \
    -v figures/noregret/values/battleship-large.pdf \
    -n "battleship (large)" \
    > data/noregret/battleship-large.json
python scripts/solve.py \
    games/battleship-huge.json \
    noregret.games.TwoPlayerZeroSumExtensiveFormGame \
    noregret.regret_minimizers.CounterfactualRegretMinimization \
    8 \
    -a \
    -v figures/noregret/values/battleship-huge.pdf \
    -n "battleship (huge)" \
    > data/noregret/battleship-huge.json
```

Solve games using OpenSpiel (C++).

```console
python scripts/open-spiel-solve.py \
    kuhn_poker \
    pyspiel.CFRSolver \
    1024 \
    -e figures/pyspiel/exploitabilities/kuhn-poker.pdf \
    > data/pyspiel/kuhn-poker.json
python scripts/open-spiel-solve.py \
    leduc_poker \
    pyspiel.CFRSolver \
    1024 \
    -e figures/pyspiel/exploitabilities/leduc-poker.pdf \
    > data/pyspiel/leduc-poker.json
python scripts/open-spiel-solve.py \
    liars_dice \
    pyspiel.CFRSolver \
    1024 \
    -e figures/pyspiel/exploitabilities/liars-dice.pdf \
    > data/pyspiel/liars-dice.json
python scripts/open-spiel-solve.py \
    'battleship(board_height=2,board_width=2,ship_sizes=[2],ship_values=[1],num_shots=3)' \
    pyspiel.CFRSolver \
    8 \
    > data/pyspiel/battleship-tiny.json
python scripts/open-spiel-solve.py \
    'battleship(board_height=3,board_width=2,ship_sizes=[2],ship_values=[1],num_shots=3)' \
    pyspiel.CFRSolver \
    8 \
    > data/pyspiel/battleship-small.json
python scripts/open-spiel-solve.py \
    'battleship(board_height=4,board_width=4,ship_sizes=[1],ship_values=[1],num_shots=2)' \
    pyspiel.CFRSolver \
    8 \
    > data/pyspiel/battleship-medium.json
python scripts/open-spiel-solve.py \
    'battleship(board_height=3,board_width=3,ship_sizes=[1;2],ship_values=[1;1],num_shots=2)' \
    pyspiel.CFRSolver \
    8 \
    > data/pyspiel/battleship-large.json
python scripts/open-spiel-solve.py \
    'battleship(board_height=3,board_width=2,ship_sizes=[2],ship_values=[1],num_shots=4)' \
    pyspiel.CFRSolver \
    8 \
    > data/pyspiel/battleship-huge.json
```

Summarize results.

```console
python scripts/summary.py \
    8 \
    data/gpugt/kuhn-poker.json \
    data/gpugt/leduc-poker.json \
    data/gpugt/liars-dice.json \
    data/gpugt/battleship-tiny.json \
    data/gpugt/battleship-small.json \
    data/gpugt/battleship-medium.json \
    data/gpugt/battleship-large.json \
    data/gpugt/battleship-huge.json \
    data/noregret/kuhn-poker.json \
    data/noregret/leduc-poker.json \
    data/noregret/liars-dice.json \
    data/noregret/battleship-tiny.json \
    data/noregret/battleship-small.json \
    data/noregret/battleship-medium.json \
    data/noregret/battleship-large.json \
    data/noregret/battleship-huge.json \
    data/pyspiel/kuhn-poker.json \
    data/pyspiel/leduc-poker.json \
    data/pyspiel/liars-dice.json \
    data/pyspiel/battleship-tiny.json \
    data/pyspiel/battleship-small.json \
    data/pyspiel/battleship-medium.json \
    data/pyspiel/battleship-large.json \
    data/pyspiel/battleship-huge.json \
    > data/summary.csv
python scripts/plot.py \
    figures/iteration-time-plot.pdf \
    figures/memory-usage-plot.pdf \
    figures/cuda-memory-usage-plot.pdf \
    < data/summary.csv
python scripts/latex.py < data/summary.csv > data/latex.tex
```

## Citing

If you use GPUGT in your research, please cite our preprint:

```bibtex
@misc{kim2024gpuacceleratedcounterfactualregretminimization,
      title={GPU-Accelerated Counterfactual Regret Minimization},
      author={Juho Kim},
      year={2024},
      eprint={2408.14778},
      archivePrefix={arXiv},
      primaryClass={cs.GT},
      url={https://arxiv.org/abs/2408.14778},
}
```
