from dataclasses import dataclass, field, KW_ONLY
from typing import Any

from cupyx.scipy.sparse import csr_matrix
from noregret.regret_minimizers import SequenceFormPolytopeRegretMinimizer
from scipy.sparse import lil_array
import cupy as cp


@dataclass
class CounterfactualRegretMinimization(SequenceFormPolytopeRegretMinimizer):
    tree_form_sequential_decision_process: Any
    _: KW_ONLY
    dimension: Any = field(init=False)
    is_time_symmetric: Any = True
    mask: Any = field(init=False)
    counterfactual_regrets: Any = field(init=False)
    behavioral_uniform_strategy: Any = field(init=False)
    behavioral_strategy: Any = field(init=False)

    def __post_init__(self):
        tfsdp = self.tree_form_sequential_decision_process
        self.dimension = len(tfsdp.sequences)

        super().__post_init__()

        self.previous_strategy = cp.array(self.previous_strategy)
        self.average_strategy = cp.array(self.average_strategy)
        self.previous_utility = cp.array(self.previous_utility)
        self.cumulative_utility = cp.array(self.cumulative_utility)

        self.mask = lil_array(
            (len(tfsdp.decision_points), len(tfsdp.sequences) - 1),
        )

        for node, j in enumerate(tfsdp.decision_points):
            for a in tfsdp.actions[j]:
                sequence = tfsdp.sequences.index((j, a)) - 1
                self.mask[node, sequence] = 1

        self.mask = csr_matrix(self.mask)
        self.counterfactual_regrets = cp.zeros(len(tfsdp.sequences) - 1)
        self.behavioral_uniform_strategy = tfsdp.behavioral_uniform_strategy()
        self.behavioral_strategy = self.behavioral_uniform_strategy.copy()

    @property
    def _floored_counterfactual_regrets(self):
        return self.counterfactual_regrets.clip(0)

    def next_strategy(self, prediction=False):
        if prediction is not False:
            raise NotImplementedError

        numerator = self._floored_counterfactual_regrets
        denominator = self.mask.T @ (self.mask @ numerator)
        normalized_denominator = denominator.copy()
        normalized_denominator[normalized_denominator == 0] = 1
        self.behavioral_strategy = cp.where(
            denominator == 0,
            self.behavioral_uniform_strategy,
            numerator / normalized_denominator,
        )
        strategy = (
            self
            .tree_form_sequential_decision_process
            .behavioral_to_sequence_form(self.behavioral_strategy)
        )

        self.strategies.append(strategy)

        return strategy

    def observe_utility(self, utility):
        super().observe_utility(utility)

        counterfactual_utilities = (
            self
            .tree_form_sequential_decision_process
            .counterfactual_utilities(self.behavioral_strategy, utility)
        )
        self.counterfactual_regrets += (
            counterfactual_utilities
            - (
                self.mask.T
                @ (
                    self.mask
                    @ (self.behavioral_strategy * counterfactual_utilities)
                )
            )
        )


@dataclass
class CounterfactualRegretMinimizationPlus(CounterfactualRegretMinimization):
    _: KW_ONLY
    gamma: Any = 1

    @property
    def _floored_counterfactual_regrets(self):
        return self.counterfactual_regrets

    def observe_utility(self, utility):
        super().observe_utility(utility)

        self.counterfactual_regrets = self.counterfactual_regrets.clip(0)
