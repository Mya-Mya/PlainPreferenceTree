from typing import List, Dict, Any
from .turns import Turn, turns_to_conversation
from itertools import product


def create_preference_samples_from_last_turn(turns: List[Turn]) -> List[Dict[str, Any]]:
    """
    Creates the preference samples from the list of turns.
    * The prompt is formed by all turns from the first to the second-to-last.
    * The chosen/rejected pairs are formed only by the last turn. 
       * The `main` property of the last turn is also used as the chosen sample.
       * The last turn should includes 1 or more rejected contents.
    
    The output is in the conversational format, defined by HuggingFace.
    For more details on the conversational format, see [HuggingFace documentation](https://huggingface.co/docs/trl/dataset_formats#dataset-formats-and-types).
    
    Parameters:
    turns (List[Turn]): The list of turns.

    Returns:
    List[Dict[str, Any]]: Samples of the preference data in the conversational format, defined by HuggingFace.
    """
    last_turn = turns[-1]
    assert last_turn.rejecteds, ValueError(
        f"At least 1 sample of rejected content is needed at the last turn."
    )
    following_turns = turns[:-1]
    prompt = turns_to_conversation(following_turns)
    chosen_contents = last_turn.chosens + [last_turn.main]
    rejected_contents = last_turn.rejecteds
    samples = []
    for chosen_content, rejected_content in product(chosen_contents, rejected_contents):
        sample = {
            "prompt": prompt,
            "chosen": [{"role": last_turn.role, "content": chosen_content}],
            "rejected": [{"role": last_turn.role, "content": rejected_content}],
        }
        samples.append(sample)
    return samples


def create_preference_samples(turns: List[Turn]) -> List[Dict[str, Any]]:
    """
    Creates the preference samples from the list of turns.
    It generates samples using the `create_preference_samples_from_last_turn`.
    
    The output is in the conversational format, defined by HuggingFace.
    For more details on the conversational format, see [HuggingFace documentation](https://huggingface.co/docs/trl/dataset_formats#dataset-formats-and-types).
    
    Parameters:
    turns (List[Turn]): The list of turns.

    Returns:
    List[Dict[str, Any]]: Samples of the preference data in the conversational format, defined by HuggingFace.
    """
    samples = []
    for i, turn in enumerate(turns):
        if turn.rejecteds:
            samples_at_i = create_preference_samples_from_last_turn(turns[: i + 1])
            samples = samples + samples_at_i
    return samples
