from functools import reduce


def getattrd(obj, name):
    """
    Same as getattr(), but allows dot notation lookup.

    Discussed in:
    http://stackoverflow.com/questions/11975781
    """
    return reduce(getattr, name.split("."), obj)
