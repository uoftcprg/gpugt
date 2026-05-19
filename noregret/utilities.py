"""Module for utilities."""
from importlib import import_module
from random import choices


def import_object(object_path):
    """Import an object from a path.

    >>> import_object('math.inf')
    inf
    >>> import_object('math')
    <module 'math' (built-in)>

    :param object_path: Import path of the object.
    :return: Imported object.
    """
    if '.' in object_path:
        path, name = object_path.rsplit('.', 1)
        obj = getattr(import_module(path), name)
    else:
        obj = __import__(object_path)

    return obj


def split(values, counts):
    """Split concatenated values.

    >>> split([0, 1, 2, 3, 4, 5], [3, 0, 1, 2])
    [[0, 1, 2], [], [3], [4, 5]]

    :param values: Values to be split.
    :param counts: Size of the partitions.
    :return: Split values.
    """
    splits = []
    begin = 0

    for cnt in counts:
        end = begin + cnt

        splits.append(values[begin:end])

        begin = end

    return splits


def sample(values, probabilities):
    """Sample a random value as per the probabilities.

    >>> sample(range(5), [0, 0, 1, 0, 0])
    2

    :param values: Values to be sampled from.
    :param probabilities: Probabilities of sampling the values.
    :return: Sampled value.
    """
    return choices(values, probabilities)[0]


def tuple_or_none(values):
    """Create a tuple of values or return ``None``.

    :param: Optional values.
    :return: Tuple or ``None``.
    """
    return None if values is None else tuple(values)
