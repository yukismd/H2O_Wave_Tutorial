"""messages.py patches for server versions 1.10"""

from .references import *


class ServerObject:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def dump(self) -> dict:
        d = {k: (v.dump() if hasattr(v, 'dump') else v) for k, v in vars(self).items()}
        return d
