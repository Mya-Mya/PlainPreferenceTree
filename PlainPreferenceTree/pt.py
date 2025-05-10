from typing import List, Dict, Any, Literal
from dataclasses import dataclass, field
from itertools import product


@dataclass
class Subnode:
    type: Literal["upvoted", "downvoted", "writing", "unrated"] = "unrated"
    content: str = ""


@dataclass
class Turn:
    role: str = ""
    main: str = ""
    subnodes: List[Subnode] = field(default_factory=list)

    def collect_subnodes_by(self, type: str) -> List[Subnode]:
        return [n for n in self.subnodes if n.type == type]

    def collect_subnotde_contents_by(self, type: str) -> List[str]:
        return [n.content for n in self.collect_subnodes_by(type)]
    
    def collect_upvoteds(self)->List[Subnode]:
        return self.collect_subnodes_by("upvoted")
    
    def collect_downvoteds(self)->List[Subnode]:
        return self.collect_subnodes_by("downvoted")
    
    def collect_writings(self)->List[Subnode]:
        return self.collect_subnodes_by("writing")
    
    def collect_unrateds(self)->List[Subnode]:
        return self.collect_subnodes_by("unrated")

    def collect_upvoted_contents(self) -> List[str]:
        return self.collect_subnotde_contents_by("upvoted")

    def collect_downvoted_contents(self) -> List[str]:
        return self.collect_subnotde_contents_by("downvoted")

    def collect_writing_contents(self) -> List[str]:
        return self.collect_subnotde_contents_by("writing")

    def collect_unrated_contents(self) -> List[str]:
        return self.collect_subnotde_contents_by("unrated")


PT = List[Turn]


def make_conversation(pt: PT) -> List[Dict[str, str]]:
    return [{"role": turn.role, "content": turn.main} for turn in pt]


def make_preferences_from_last_turn(turns: List[Turn]) -> List[Dict[str, Any]]:
    last_turn = turns[-1]
    assert last_turn.collect_downvoted_contents(), ValueError(
        f"At least 1 sample of rejected content is required at the last turn."
    )
    following_turns = turns[:-1]
    prompt = make_conversation(following_turns)
    chosen_contents = last_turn.collect_upvoted_contents() + [last_turn.main]
    rejected_contents = last_turn.collect_downvoted_contents()
    samples = []
    for chosen_content, rejected_content in product(chosen_contents, rejected_contents):
        sample = {
            "prompt": prompt,
            "chosen": [{"role": last_turn.role, "content": chosen_content}],
            "rejected": [{"role": last_turn.role, "content": rejected_content}],
        }
        samples.append(sample)
    return samples


def make_preferences(turns: List[Turn]) -> List[Dict[str, Any]]:
    samples = []
    for i, turn in enumerate(turns):
        if turn.collect_downvoted_contents():
            samples_at_i = make_preferences_from_last_turn(turns[: i + 1])
            samples = samples + samples_at_i
    return samples
