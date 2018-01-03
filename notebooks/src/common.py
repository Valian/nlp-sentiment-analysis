import hashlib
from _hashlib import HASH
from typing import Iterable, Dict


def _to_hashable(val):
    if isinstance(val, HASH):
        return val.digest()
    if not isinstance(val, bytes):
        return str(val).encode()
    return val


def get_hash_of_array(arr: Iterable) -> HASH:
    h = hashlib.md5()
    for item in arr:
        h.update(_to_hashable(item))
    return h


def get_hash_of_dict(d: Dict) -> HASH:
    h = hashlib.md5()
    for key in sorted(d.keys()):
        val = d[key]
        h.update(_to_hashable(key))
        h.update(_to_hashable(val))
    return h
