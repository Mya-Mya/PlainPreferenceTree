from typing import List, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str = ""
    main: str = ""
    chosens: List[str] = field(default_factory=list)
    rejecteds: List[str] = field(default_factory=list)


def collect_lines_until_empty(lines: List[str]) -> Tuple[str, List[str], int]:
    """
    Collects lines from the input list until an empty string is encountered,
    and returns the collected lines, the remaining lines, and the number of collected lines.

    Parameters:
    lines (List[str]): A list of string lines to be collected.

    Returns:
    Tuple[str, List[str], int]:
        - str: A string formed by joining the collected lines with newline characters.
        - List[str]: The remaining lines in the list after collection is complete.
        - int: The number of lines collected.
    """
    collected_lines = []
    nlines = 0
    while lines and lines[0] != "":
        collected_lines.append(lines.pop(0))
        nlines += 1
    collected = "\n".join(collected_lines)
    return collected, lines, nlines


CHANGE_ROLE = {"user": "assistant", "assistant": "user"}


def pptraw_to_turns(raw: str) -> List[Turn]:
    """
    Parses the text in Plain-Preference-Tree (PPT) format into the list of `Turn`.

    Parameters:
    raw (str): A PPT format text.

    Returns:
    List[Turn]: The parsed data, the list of `Turn` objects.
    """
    turns: List[Turn] = []
    currnt_role = "user"

    lines = raw.splitlines()
    original_nrow = len(lines)
    while lines:
        turn = Turn(role=currnt_role)
        turn.main, lines, _ = collect_lines_until_empty(lines)

        collecting_other_choices = True
        while collecting_other_choices and lines:
            assert lines.pop(0) == "", ValueError(
                f"Line {original_nrow-len(lines)} should be empty."
            )
            if lines[0] == "":
                # Go to next turn.
                lines.pop(0)
                collecting_other_choices = False
            else:
                # Collect chosen or rejected samples for the current turn.
                read, lines, nlines = collect_lines_until_empty(lines)
                sign = read[:2]
                content = read[2:]
                if sign == "+ ":
                    turn.chosens.append(content)
                elif sign == "- ":
                    turn.rejecteds.append(content)
                else:
                    raise ValueError(
                        f"Line {original_nrow-len(lines)+nlines} should start with either `+ ` or `- `."
                    )
        turns.append(turn)
        currnt_role = CHANGE_ROLE[currnt_role]
    return turns


def turns_to_conversation(turns: List[Turn]) -> List[Dict[str, str]]:
    """
    Converts the list of turns into the conversational format, defined by HuggingFace.
    For more details on the conversational format, see [the HuggingFace documentation](https://huggingface.co/docs/trl/dataset_formats#dataset-formats-and-types).

    Parameters:
    turns (List[Turn]): The list of turns.

    Returns:
    List[Dict[str, str]]: The conversation data in the conversational format, defined by HuggingFace.
    """
    return [{"role": turn.role, "content": turn.main} for turn in turns]


def turns_to_pptraw(turns: List[Turn]) -> str:
    lines = []
    put = lines.append
    for turn in turns:
        put(turn.main)
        put("")
        for chosen_content in turn.chosens:
            put("+ " + chosen_content)
            put("")
        for rejected_content in turn.rejecteds:
            put("- " + rejected_content)
            put("")
        put("")
    pptraw = "\n".join(lines)
    return pptraw
