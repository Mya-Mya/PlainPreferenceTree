from gradio import Blocks, Button, Textbox, JSON
from PlainPreferenceTree.gradiostreaming import Inferencer, complete_writings, loads
from PlainPreferenceTree.dummyinferencer import DummyInferencer

inferencer: Inferencer = DummyInferencer()

def on_generate_press(ppt:str):
    return complete_writings(
        pptv1=ppt,
        inferencer=inferencer
    )

def on_showpt_press(ppt:str):
    return loads(ppt)

with Blocks(fill_width=True, fill_height=True, title="PlainPreferenceTree") as app:
    ppt_textbox = Textbox(value="", lines=20, max_lines=40, show_label=False)
    generate_button = Button(value="Submit", variant="primary")
    pt_json = JSON(value=[], label="Preference Tree")
    showpt_button = Button(value="Show Preference Tree")

    generate_button.click(fn=on_generate_press, inputs=[ppt_textbox], outputs=[ppt_textbox])
    showpt_button.click(fn=on_showpt_press, inputs=[ppt_textbox], outputs=[pt_json])

app.launch()
