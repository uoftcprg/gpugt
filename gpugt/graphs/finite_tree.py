""":mod:`gpugt.graphs.finite_tree` defines a finite tree."""

from collections.abc import Hashable
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Generic, TypeVar

from gpugt.collections2 import FrozenOrderedMapping, FrozenOrderedSet

_V = TypeVar('_V', bound=Hashable)


@dataclass(frozen=True)
class FiniteTree(Generic[_V]):
    """A representation of a finite tree.

    :param vertices: A finite set of vertices.
    :param root: The root.
    :param leaves: A finite set of leaves.
    :param parents: The parents.
    """

    vertices: FrozenOrderedSet[_V]
    """A finite set of vertices."""
    root: _V
    """The root."""
    leaves: FrozenOrderedSet[_V]
    """A finite set of leaves (vertices without any child)."""
    parents: FrozenOrderedMapping[_V, _V]
    """The parents (value) for each child (key)."""

    def __post_init__(self) -> None:
        if self.root not in self.vertices:
            raise ValueError('root not a member of nodes')
        elif not self.leaves <= self.vertices:
            raise ValueError('leaves not a subset of vertices')
        elif self.parents.keys() != self.non_roots:
            raise ValueError('parents not defined for non-roots')
        elif set(self.parents.values()) != self.internal_vertices:
            raise ValueError('non-parent internal vertex')

    @cached_property
    def non_roots(self) -> FrozenOrderedSet[_V]:
        """Return a finite set of non-roots.

        :return: A finite set of non-roots.
        """
        return FrozenOrderedSet(self.vertices - {self.root})

    @cached_property
    def internal_vertices(self) -> FrozenOrderedSet[_V]:
        """Return a finite set of internal vertices (i.e. non-leaves).

        :return: A finite set of internal vertices.
        """
        return FrozenOrderedSet(self.vertices - self.leaves)

    @cached_property
    def children(self) -> FrozenOrderedMapping[_V, FrozenOrderedSet[_V]]:
        """Return finite sets of children (value) for each parent (key).

        :return: Finite sets of children for each parent.
        """
        children = defaultdict(list)

        for child in self.non_roots:
            children[self.parents[child]].append(child)

        keys = children.keys()
        values = map(FrozenOrderedSet, children.values())

        return FrozenOrderedMapping(zip(keys, values))
