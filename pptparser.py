from typing import List
from abc import ABC, abstractmethod
from .pt import Turn, PT


class PPTParser(ABC):
    @abstractmethod
    def loads(self, text: str) -> PT:
        pass

    @abstractmethod
    def dumps(self, pt: PT) -> str:
        pass
