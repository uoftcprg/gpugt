from collections.abc import Callable
from functools import cache, wraps
from typing import Any, cast, TypeVar

_C = TypeVar('_C', bound=Callable[..., Any])


def cached_method(method: _C) -> _C:
    lookup = dict[int, Any]()

    @cache
    def wrapper_0(self_id: int, *args: Any, **kwargs: Any) -> Any:
        return method(lookup[self_id], *args, **kwargs)

    @wraps(method)
    def wrapper_1(self: Any, *args: Any, **kwargs: Any) -> Any:
        self_id = id(self)
        lookup[self_id] = self

        return wrapper_0(self_id, *args, **kwargs)

    return cast(_C, wrapper_1)
