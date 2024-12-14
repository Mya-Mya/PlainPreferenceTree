from abc import ABC, abstractmethod
from .pt import PT


class PPTParser(ABC):
    @abstractmethod
    def loads(self, text: str) -> PT:
        pass

    @abstractmethod
    def dumps(self, pt: PT) -> str:
        pass
