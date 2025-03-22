from typing import List, Dict, Iterator
from time import sleep

from .gradiostreaming import Inferencer


class DummyInferencer(Inferencer):
    def start(
            self,
            context: List[Dict[str, str]],
            role: str,
            initial_text: str
    ) -> Iterator[str]:
        for _ in range(10):
            sleep(0.3)
            yield "X"
    def stop(self):
        pass