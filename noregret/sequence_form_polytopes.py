"""Module for sequence-form polytopes."""
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from ordered_set import OrderedSet
from scipy.sparse import lil_array

from noregret.kernels import Kernel


@dataclass
class SequenceFormPolytope:
    """Class for sequence-form polytopes.

    Any vector in behavioral form is of length equal to the number of
    non-empty sequences. Any vector in sequence form is of length equal
    to the number of sequences.
    """
    kernel: Kernel
    """Kernel."""
    actions: dict[str, OrderedSet[str]]
    """Actions."""
    parent_sequences: dict[str, tuple[str, str] | None]
    """Parent sequences for decision points."""
    constraint_matrix: Any = field(init=False)
    """Constraint matrix."""
    constraint_vector: Any = field(init=False)
    """Constraint vector."""
    _A: Any = field(init=False)
    _B: Any = field(init=False)
    _R: Any = field(init=False)
    _C: Any = field(init=False)
    _L: list[Any] = field(default_factory=list, init=False)
    _L_prime: list[Any] = field(default_factory=list, init=False)

    def __post_init__(self):
        if self.decision_points != self.parent_sequences.keys():
            raise ValueError('inconsistent decision points')

        np = self.kernel.numpy
        scipy = self.kernel.scipy
        A = lil_array((self.row_count, self.column_count))
        B = lil_array((self.row_count, self.column_count))
        A[0, 0] = 1

        for j in self.decision_points:
            p_j = self.parent_sequences[j]
            r = self.row(j)
            c = self.column(p_j)
            B[r, c] = 1

            for a in self.actions[j]:
                c = self.column((j, a))
                A[r, c] = 1

        self._A = scipy.sparse.csr_array(A)
        self._B = scipy.sparse.csr_array(B)
        self.constraint_matrix = self._A - self._B
        self.constraint_vector = self.kernel.standard_basis(self.row_count, 0)
        R = []
        C = []

        for j, a in self.non_empty_sequences:
            R.append(self.row(j))
            C.append(self.column((j, a)))

        self._R = np.ascontiguousarray(np.array(R))
        self._C = np.ascontiguousarray(np.array(C))
        children = defaultdict(list)

        for j, p_j in self.parent_sequences.items():
            parent = None if p_j is None else p_j[0]

            children[parent].append(j)

        J = [None]

        while J:
            J_prime = []
            L = []
            L_prime = OrderedSet()

            for j in J:
                J_prime.extend(children[j])
                L.append(self.row(j))

                if j is not None:
                    L_prime.add(self.column(self.parent_sequences[j]))

            J = J_prime

            self._L.append(np.ascontiguousarray(np.array(L)))
            self._L_prime.append(np.ascontiguousarray(np.array(L_prime)))

    @cached_property
    def decision_points(self):
        """Return decision points.

        :return: Decision points.
        """
        return OrderedSet(self.actions.keys())

    @cached_property
    def non_empty_sequences(self):
        """Return non-empty sequences.

        :return: Non-empty sequences.
        """
        sequences = OrderedSet()

        for j in self.decision_points:
            for a in self.actions[j]:
                sequences.add((j, a))

        return sequences

    @property
    def row_count(self):
        """Return the number of rows in the constraint matrix.

        :return: Number of rows.
        """
        return len(self.decision_points) + 1

    @property
    def column_count(self):
        """Return the number of columns in the constraint matrix.

        :return: Number of columns.
        """
        return len(self.non_empty_sequences) + 1

    def row(self, decision_point):
        """Return the corresponding row of a given decision point in the
        constraint matrix.

        :param decision_point: Decision point.
        :return: Corresponding row.
        """
        if decision_point is None:
            r = 0
        else:
            r = self.decision_points.index(decision_point) + 1

        return r

    def column(self, sequence):
        """Return the corresponding column of a given sequence in the
        constraint matrix.

        :param sequence: sequence.
        :return: Corresponding column.
        """
        if sequence is None:
            c = 0
        else:
            c = self.non_empty_sequences.index(sequence) + 1

        return c

    @cached_property
    def behavioral_form_uniform_strategy(self):
        """Return the uniform strategy in behavioral form.

        :return: The uniform strategy in behavioral form.
        """
        return ((1 / self._A.sum(1)).T @ self._A).ravel()[1:]

    def to_sequence_form(self, behavioral_strategy):
        """Convert a strategy (in behavioral form) to sequence form.

        :param behavioral_strategy: Strategy in behavioral form.
        :return: Strategy in sequence form.
        """
        if behavioral_strategy.shape != (len(self.non_empty_sequences),):
            raise ValueError('invalid strategy shape')

        A = self._A.copy()
        A[self._R, self._C] = behavioral_strategy
        A_T = A.T

        for L in self._L[1:]:
            A_T[:, L] = A_T[:, L].multiply((self._B[L] @ A_T).sum(1).T)

        return A_T.sum(1).ravel()

    def _counterfactual_utilities_or_regrets(
            self,
            behavioral_strategy,
            utility,
            normalize,
    ):
        if behavioral_strategy.shape != (len(self.non_empty_sequences),):
            raise ValueError('invalid strategy shape')
        elif utility.shape != (self.column_count,):
            raise ValueError('invalid utility shape')

        A = self._A.copy()
        A[self._R, self._C] = behavioral_strategy
        A_T = A.T
        utility = self.kernel.scipy.sparse.csr_array(utility)

        for L in reversed(self._L[1:]):
            utility += utility @ A_T[:, L] @ self._B[L]

        utility = utility.toarray().ravel()

        if normalize:
            utility -= utility @ A_T @ self._A

        return utility[1:]

    def counterfactual_utilities(self, behavioral_strategy, utility):
        """Calculate the counterfactual utilities given a behavioral
        strategy and utility.

        :param behavioral_strategy: Strategy in behavioral form.
        :param utility: Utility.
        :return: Counterfactual utilities.
        """
        return self._counterfactual_utilities_or_regrets(
            behavioral_strategy,
            utility,
            False,
        )

    def counterfactual_regrets(self, behavioral_strategy, utility):
        """Calculate the counterfactual regrets given a behavioral
        strategy and utility.

        :param behavioral_strategy: Strategy in behavioral form.
        :param utility: Utility.
        :return: Counterfactual regrets.
        """
        np = self.kernel.numpy

        if np.isscalar(utility):
            r = np.zeros(len(self.non_empty_sequences))
        else:
            r = self._counterfactual_utilities_or_regrets(
                behavioral_strategy,
                utility,
                True,
            )

        return r

    def normalize(self, vector):
        """L1-normalize a given vector decision-point-wise.

        :param vector: Vector.
        :return: Normalized vector.
        """
        if vector.shape != (len(self.non_empty_sequences),):
            raise ValueError('invalid vector shape')

        A = self._A.copy()
        A[self._R, self._C] = vector
        d = A.sum(1).ravel()
        r = d == 0
        A[r] = self._A[r]
        d[r] = A[r].sum(1).ravel()
        v = ((1 / d) @ A).ravel()

        return v[1:]

    def best_response_value(self, utility):
        """Calculate the best response value given a utility.

        The implementation requires the sparse matrix implementation to
        not prune explicit zeros in the Hadamard product.

        :param utility: Utility.
        :return: Best response value.
        """
        if utility.shape != (self.column_count,):
            raise ValueError('invalid shape')

        u = utility.copy()

        for L_prime, L in zip(self._L_prime[:0:-1], self._L[:0:-1]):
            v = self._A[L].multiply(u).max(1, explicit=True).T @ self._B[L]
            u[L_prime] += v.reshape(1, -1)[0, L_prime].toarray().ravel()

        return u[0]

    def worst_response_value(self, utility):
        """Calculate the worst response value given a utility.

        The implementation requires the sparse matrix implementation to
        not prune explicit zeros in the Hadamard product.

        :param utility: Utility.
        :return: Worst response value.
        """
        if utility.shape != (self.column_count,):
            raise ValueError('invalid shape')

        u = utility.copy()

        for L_prime, L in zip(self._L_prime[:0:-1], self._L[:0:-1]):
            v = self._A[L].multiply(u).min(1, explicit=True).T @ self._B[L]
            u[L_prime] += v.reshape(1, -1)[0, L_prime].toarray().ravel()

        return u[0]
