from dataclasses import dataclass, field
from typing import Any

from cupyx.scipy.sparse import csr_matrix
from scipy.sparse import lil_array
import cupy as cp
import numpy as np


@dataclass
class TreeFormSequentialDecisionProcess:
    tree_form_sequential_decision_process: Any
    level_sources: Any = field(init=False, default_factory=list)
    level_sequences: Any = field(init=False, default_factory=list)
    level_parent_sequences: Any = field(init=False, default_factory=list)
    graph: Any = field(init=False)
    graph2: Any = field(init=False)
    behavioral: Any = field(init=False)
    behavioral2: Any = field(init=False)
    counterfactual: Any = field(init=False)

    def __post_init__(self):
        assert self.sequences[0] == ()

        queue = [self.nodes[0]]

        while queue:
            sources = []
            sequences = []
            parent_sequences = []
            next_queue = []

            for p in queue:
                sources.append(self.nodes.index(p))

                match self.node_types[p]:
                    case self.NodeType.DECISION_POINT:
                        parent_sequence = self.sequences.index(
                            self.parent_sequences[p],
                        )

                        for a in self.actions[p]:
                            sequences.append(self.sequences.index((p, a)))
                            parent_sequences.append(parent_sequence)
                            next_queue.append(self.transitions[p, a])
                    case self.NodeType.OBSERVATION_POINT:
                        for s in self.signals[p]:
                            next_queue.append(self.transitions[p, s])

            self.level_sources.append(cp.array(sources, np.long))
            self.level_sequences.append(cp.array(sequences, np.long))
            self.level_parent_sequences.append(
                cp.array(parent_sequences, np.long),
            )

            queue = next_queue

        self.graph = lil_array((len(self.nodes), len(self.nodes)))
        self.graph2 = lil_array((len(self.nodes), len(self.sequences)))

        for source, p in enumerate(self.nodes):
            match self.node_types[p]:
                case self.NodeType.DECISION_POINT:
                    for a in self.actions[p]:
                        sequence = self.sequences.index((p, a))
                        self.graph2[source, sequence] = 1

                    events = self.actions
                case self.NodeType.OBSERVATION_POINT:
                    events = self.signals

            for e in events[p]:
                target = self.nodes.index(self.transitions[p, e])
                self.graph[source, target] = 1

        self.graph = csr_matrix(self.graph)
        self.graph2 = csr_matrix(self.graph2)
        sources = []
        targets = []
        sequences = []

        for sequence, j_a in enumerate(self.sequences[1:]):
            j, _ = j_a
            source = self.nodes.index(j)
            target = self.nodes.index(self.transitions[j_a])
            sequence += 1

            sources.append(source)
            targets.append(target)
            sequences.append(sequence)

        self.behavioral = (
            cp.array(sources, np.long),
            cp.array(targets, np.long),
        )
        self.behavioral2 = (
            cp.array(sources, np.long),
            cp.array(sequences, np.long),
        )
        self.counterfactual = cp.array(
            list(
                map(
                    self.nodes.index,
                    map(self.transitions.get, self.sequences[1:]),
                ),
            ),
        )

    def __getattr__(self, name):
        return getattr(self.tree_form_sequential_decision_process, name)

    def behavioral_uniform_strategy(self):
        strategy = [None] * (len(self.sequences) - 1)

        for sequence, j_a in enumerate(self.sequences[1:]):
            j, _ = j_a
            strategy[sequence] = 1 / len(self.actions[j])

        return cp.array(strategy)

    def behavioral_best_response(self, utility):
        V = [0] * len(self.nodes)
        node = len(self.nodes)

        for p in reversed(self.nodes):
            node -= 1

            match self.node_types[p]:
                case self.NodeType.DECISION_POINT:
                    V[node] = max(
                        (
                            utility[self.sequences.index((p, a))]
                            + V[self.nodes.index(self.transitions[p, a])]
                        )
                        for a in self.actions[p]
                    )
                case self.NodeType.OBSERVATION_POINT:
                    V[node] = sum(
                        V[self.nodes.index(self.transitions[p, s])]
                        for s in self.signals[p]
                    )

        return NotImplemented, V[0]

    def sequence_form_best_response(self, utility):
        _, value = self.behavioral_best_response(utility)

        return NotImplemented, value

    def behavioral_to_sequence_form(self, behavioral_strategy):
        strategy = cp.empty(len(self.sequences))
        strategy[0] = 1
        strategy[1:] = behavioral_strategy

        for sequences, parent_sequences in zip(
                self.level_sequences,
                self.level_parent_sequences,
        ):
            strategy[sequences] *= strategy[parent_sequences]

        return strategy

    def counterfactual_utilities(self, behavioral_strategy, utility):
        graph = self.graph.copy()
        graph[self.behavioral] = behavioral_strategy
        graph2 = self.graph2.copy()
        graph2[self.behavioral2] = behavioral_strategy
        V = cp.zeros(len(self.nodes))

        for sources in reversed(self.level_sources):
            V[sources] = graph[sources] @ V + graph2[sources] @ utility

        return utility[1:] + V[self.counterfactual]
