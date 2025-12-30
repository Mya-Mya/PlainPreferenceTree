from typing import Iterator, List, Dict, Optional
from threading import Thread
from asyncio import Event
from ..PlainPreferenceTree.gradiostreaming import Inferencer
from transformers import GenerationMixin, AutoTokenizer, TextIteratorStreamer, StoppingCriteria, GenerationConfig
from torch import inference_mode, tensor


class GenerationStopper(StoppingCriteria):
    def __init__(self, event: Event):
        self.event = event

    def __call__(self, *args, **kwargs):
        return self.event.is_set()


class TransformerLLMInferencer(Inferencer):
    def __init__(
            self,
            llm: GenerationMixin,
            tokenizer: AutoTokenizer,
            truncate_len: int,
            generation_config: Optional[GenerationConfig] = None
    ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.truncate_len = truncate_len
        self.generation_config = generation_config
        self.stopper: Optional[Event] = None
        self.thread: Optional[Thread] = None

    def stop(self):
        if self.stopper:
            self.stopper.set()
            self.thread.join()
            self.stopper = None
            self.thread = None

    def start(
            self,
            context: List[Dict[str, str]],
            role: str,
            initial_text: str
    ) -> Iterator[str]:
        self.stop()
        current_context = context
        input_ids_has_ending = self.tokenizer.apply_chat_template(current_context, add_generation_prompt=False)
        input_ids = input_ids_has_ending[:-self.truncate_len]
        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        self.stopper = Event()
        stopping_criteria = [GenerationStopper(self.stopper)]

        with inference_mode():
            kwargs = dict(
                inputs=tensor(input_ids).to(self.llm.device)[None],
                generation_config=self.generation_config,
                streamer=streamer,
                stopping_criteria=stopping_criteria,
                use_cache=True
            )
            self.thread = Thread(
                target=self.llm.generate,
                kwargs=kwargs
            )
            self.thread.start()
        return streamer
