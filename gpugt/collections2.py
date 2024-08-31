""":mod:``gpugt.collections2` defines the collections-related
utilities.
"""

from collections.abc import Hashable, Iterable, Iterator, Mapping, Set
from itertools import repeat
from typing import Any, TypeVar

_KT = TypeVar('_KT', bound=Hashable)
_VT = TypeVar('_VT')
_T = TypeVar('_T', bound=Hashable)


class FrozenOrderedMapping(Mapping[_KT, _VT]):
    """An implementation of frozen ordered mappings.

    This class can be instantiated exactly like a ``dict``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.__data: dict[_KT, _VT] = dict(*args, **kwargs)

    def __getitem__(self, key: _KT) -> _VT:
        return self.__data[key]

    def __iter__(self) -> Iterator[_KT]:
        return iter(self.__data)

    def __len__(self) -> int:
        return len(self.__data)

    def __repr__(self) -> str:
        return repr(self.__data)


class FrozenOrderedSet(Set[_T]):
    """An implementation of frozen ordered sets.

    This class can be instantiated exactly like a ``set``.

    :param iterable: An iterable of members.
    """

    def __init__(self, iterable: Iterable[_T] = ()) -> None:
        self.__data = dict(zip(iterable, repeat(None)))

    def __contains__(self, o: object) -> bool:
        return o in self.__data

    def __iter__(self) -> Iterator[_T]:
        return iter(self.__data)

    def __len__(self) -> int:
        return len(self.__data)

    def __eq__(self, value: object) -> bool:
        return set(self.__data) == value

    def __repr__(self) -> str:
        return '{' + ', '.join(map(repr, self.__data.keys())) + '}'
