from typing import Iterator, Generator, List, Dict
from .pptparserv1 import PPTParserV1
from .pt import PT, Subnode, make_conversation
from abc import ABC, abstractmethod

parser = PPTParserV1()
dumps = parser.dumps
loads = parser.loads


class Inferencer(ABC):
    @abstractmethod
    def start(
            self,
            context: List[Dict[str, str]],
            role: str,
            initial_text: str
    ) -> Iterator[str]:
        pass
    @abstractmethod
    def stop(self):
        pass


def complete_writings(
        pptv1: str,
        inferencer: Inferencer
) -> Generator[str, None, None]:
    pt = loads(pptv1)
    for k, turn in enumerate(pt):
        writings = turn.collect_writings()
        context = make_conversation(pt[:k])
        for writing in writings:
            streamer = inferencer.start(
                context=context,
                role=turn.role,
                initial_text=writing.content
            )
            for flushed in streamer:
                writing.content += flushed
                yield dumps(pt)
    yield dumps(pt)
    return
