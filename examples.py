from .turns import pptraw_to_turns, turns_to_conversation, turns_to_pptraw
from .preferences import create_preference_samples
from pprint import pprint

pptraw = """Hello


Hello! I'm glad to see you.

- Oh, sorry, I am still sleepy. Please call me later.

+ Hello! It is nice to see you.

- Good bye. I don't want to talk with you.


Yea. I am very happy. 


You look that you slept well, didn't you?


You are right. Today, I'd like to do something special.
Do you have any recommendations?


Of course. How about reading some books stacking at the corner of your room?

- No. You should just sleep.

+ Yes. Let's go shopping!
"""


def perform_example():
    print("Plain-Preference-Text Format Text:")
    print(pptraw)
    print()

    print("Turns:")
    turns = pptraw_to_turns(pptraw)
    pprint(turns)
    print()

    print("Conversation:")
    conversation = turns_to_conversation(turns)
    pprint(conversation)
    print()

    print("Preference Samples:")
    preference_samples = create_preference_samples(turns)
    pprint(preference_samples)
    print()

    print("Plain-Preference-Tree Format Text (Restored from `turns`):")
    pptraw_restored = turns_to_pptraw(turns)
    print(pptraw_restored)
